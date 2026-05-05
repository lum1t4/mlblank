from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timedelta
import gc
import hashlib
import io
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Union
from urllib.request import urlretrieve

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mlblank.core.pytorch import (
    LOCAL_RANK,
    RANK,
    WORLD_SIZE,
    convert_optimizer_state_dict_to_fp16,
    device_memory_clear,
    intersect_dicts,
    model_deparallel,
    rank_zero_only,
    set_seeds,
)
from mlblank.core.utils import LOGGER, IterableSimpleNamespace

CACHE_DIR = "~/.cache/model_registry"
CACHE_DIR = Path(CACHE_DIR).expanduser()


EXT_PYTORCH = {'.pt', '.pth', '.bin'}
EXT_SAFETENSOR = {".safetensors"}
WANDB_AVAILABLE = False


# --- Registry Singleton ---
class ModelRegistry:
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_model(
        cls,
        model_name: str,
        model_class: Type | Callable,
        model_config: Any,
        model_weights: Union[str, None] = None,
    ):
        """Register a model in the registry."""
        if model_name in cls._registry:
            raise ValueError(f"Model '{model_name}' is already registered.")

        cls._registry[model_name] = {"class": model_class, "config": model_config, "weights": model_weights}

    @classmethod
    def get_model_entry(cls, name: str) -> Dict[str, Any]:
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry.")
        return cls._registry[name]


class Tracker:
    def __init__(self, config: dict = None):
        pass

    @rank_zero_only
    def log(self, x: dict, step: int | None = None):
        pass

    @rank_zero_only
    def log_model(self, checkpoint: Path, aliases: List[str] = ["last"]):
        pass


class TrackerRegistry:
    _registry: dict[str, Tracker] = {}

    @classmethod
    def register_tracker(cls, name: str, tracker: Tracker):
        cls._registry[name] = tracker

    @classmethod
    def list_trackers(cls):
        return list(cls._registry.keys())
    
    @classmethod
    def get_tracker(cls, name: str):
        return cls._registry[name]


def load_tracker(name: str | None, config: dict):
    if name is None or RANK not in {-1, 0}:
        return Tracker(config)
    return TrackerRegistry.get_tracker(name)(config)



try:
    import wandb
    WANDB_AVAILABLE = True

    class WandbTracker(Tracker):
        def __init__(self, config: dict):
            super().__init__(config)
            if not WANDB_AVAILABLE:
                raise ImportError("wandb is not available. Please install it to use WandbTracker.")

            self.run = wandb.init(
                project=config['project'],
                name=config['name'],
                config=config,
                allow_val_change=True
            ) if RANK in {-1, 0} else None

        @rank_zero_only
        def log(self, x, step: int | None = None):
            self.run.log(x, step=step)

        @rank_zero_only
        def log_model(self, checkpoint, aliases = ["last"]):
            artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
            artifact.add_file(checkpoint, name=checkpoint.name)
            wandb.run.log_artifact(artifact, aliases=aliases)

    if RANK in {-1, 0}:
        TrackerRegistry.register_tracker("wandb", WandbTracker)

except ImportError:
    wandb = None



def retrive_weights(path_or_url: str) -> str:
    """Download weights if it's a URL, otherwise verify path exists."""
    # remote
    if path_or_url.startswith("http"):

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(path_or_url).suffix
        identifier = hashlib.sha256(path_or_url.encode()).hexdigest()[:16]
        local_path = CACHE_DIR / f"{identifier}{suffix}"
        if not os.path.exists(local_path):
            LOGGER.info(f"Downloading weights from {path_or_url} to {local_path.as_posix()}...")
            urlretrieve(path_or_url, local_path)
        return local_path
    
    # local
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Weight file not found: {path_or_url}")
    return Path(path_or_url)


def load_weights(model: torch.nn.Module, weight_path: Path, strict: bool = False) -> torch.nn.Module:
    if weight_path.suffix in EXT_PYTORCH:
        ckpt = torch.load(weight_path, map_location="cpu")
    elif weight_path.suffix in EXT_SAFETENSOR:
        from safetensors.torch import load_file
        ckpt = load_file(weight_path, device="cpu")
    else:
        raise ValueError(f"Unsupported weight file format: {weight_path}")
    
    ckpt = ckpt['model'] if 'model' in ckpt else ckpt
    csd = intersect_dicts(ckpt, model.state_dict())  # intersect
    model.load_state_dict(csd, strict=strict)  # load

    # Warn only if there is a mismatch (intention if loading from a pretrained model)
    if RANK in {-1, 0}:
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights from {weight_path}")
    return model


def load_model(name: str, config: dict = {}, strict: bool = False, weights: str = None) -> torch.nn.Module:
    entry = ModelRegistry.get_model_entry(name)
    model_class = entry["class"]
    model_config = entry["config"]
    model_weights = weights if weights is not None else entry.get("weights", None)

    for k, v in config.items():
        if hasattr(model_config, k):
            setattr(model_config, k, v)
    
    model = model_class(model_config)
    if model_weights is not None:
        model = load_weights(model, retrive_weights(model_weights), strict=strict)

    return model


class TrainContext:
    config: IterableSimpleNamespace
    model: nn.Module | nn.parallel.DistributedDataParallel
    device: torch.device
    tracker: Tracker
    save_dir: Path
    fitness: float
    train_dataset: Dataset
    valid_dataset: Dataset | None
    train_loader: DataLoader
    valid_loader: DataLoader | None
    metrics: dict[str, Any] = {}

    # Initial iteration/epoch (it is greater than zero when resuming the run)
    start_iter: int = 0
    # Best iteration / epoch
    best_iter: int = 0
    # Current iteration / epoch
    curr_iter: int = None
    # Flag that changes when triggering early stopping
    stop: bool = False
    # Run name (logging)
    name: str | None = None
    # Enable automatic mixed precision
    mixed_precision: bool = False
    # Precision Type
    precision: torch.dtype = torch.float32

    def __init__(self, config: IterableSimpleNamespace):
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        # Set seed
        set_seeds(config.seed, deterministic=config.deterministic)

        self.config = config
        self.device = torch.device(config.device)
        self.save_dir = Path(config.save_dir)
        self.fitness = float("-inf") if config.mode == "max" else float("inf")
        self.tracker = load_tracker(name=self.config.tracker, config=vars(self.config))

        # Distributed init
        if WORLD_SIZE > 1:
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=10800)) # 3 hours
            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device(f"cuda:{LOCAL_RANK}")

        
        # Checks and patches on config
        if self.device.type in {"cpu", "mps"}:
            self.config.workers = 0

        self.configure_precision(config)


    def configure_precision(self, config: IterableSimpleNamespace):
        self.mixed_precision = getattr(config, "mixed_precision", True)
        # Checks and patches on config
        if self.device.type in {"cpu", "mps"}:
            self.mixed_precision = False

        value = getattr(config, "precision", "float32")
        if value in {"float32", "FP32"}:
                self.precision = torch.float32
            
        if value in {"bfloat16"}:
                self.precision = torch.bfloat16


    @property
    def weights_dir(self) -> Path:
        name = self.config.name if self.config.name else self.config.model
        w = self.save_dir / name / 'weights'
        w.mkdir(parents=True, exist_ok=True)
        return w
    
    @property
    def plot_dir(self) -> Path:
        name = self.config.name if self.config.name else self.config.model
        p = self.save_dir / name / 'plots'
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def last_checkpoint(self) -> Path:
        return self.weights_dir / "last.pth"
    
    @property
    def best_checkpoint(self) -> Path:
        return self.weights_dir / 'best.pth'
    
    @property
    def current_checkpoint(self) -> Path:
        return self.weights_dir / f"epoch_{self.curr_iter}.pt"


    def iteration_end(self):
        gc.collect()
        device_memory_clear(self.device)



    @rank_zero_only
    def checkpointing(self):
        buffer = io.BytesIO()
        model = model_deparallel(self.model) if WORLD_SIZE > 1 else self.model
        optim = convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict()))

        torch.save({
            "epoch": self.curr_iter,
            "model": model.state_dict(),
            "optimizer": optim,
            "metrics": self.metrics,
            "config": vars(self.config),
            "date": datetime.now().isoformat(),
        }, buffer)

        ckpt = buffer.getvalue()
        self.last_checkpoint.write_bytes(ckpt)
        if self.curr_iter == self.best_iter:
            self.best_checkpoint.write_bytes(ckpt)
            self.tracker.log_model(self.best_checkpoint, aliases=["best"])
        if self.config.save_period > 0 and (self.curr_iter + 1) % self.config.save_period == 0:
            self.current_checkpoint.write_bytes(ckpt)
        return self


    def resume(self):
        """Load pretrained or resume from checkpoint """
        if self.config.weights is None:
            return self

        weights = Path(self.config.weights)
        
        if not weights.exists():
            LOGGER.warning(f"Could not find specified weights at {weights}")
            return self
        
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        if self.config.resume:
            LOGGER.info("Resuming training")
            msd = ckpt["model"]

            if isinstance(ckpt["optimizer"], tuple):
                ckpt["optimizer"] = ckpt["optimizer"][0]
            
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_iter = ckpt["epoch"] + 1
            LOGGER.info(f"Resuming training from epoch {self.start_iter}")
        else:
            msd = ckpt
        csd = intersect_dicts(msd, self.model.state_dict())  # intersect
        self.model.load_state_dict(csd, strict=False)  # load
        if RANK in {-1, 0}:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")
        return self


    def early_stopping(self):
        x = self.metrics[self.config.monitor]
        y = self.fitness
        if (self.config.mode == "max" and x >= y) or (self.config.mode == "min" and x <= y):
            self.fitness = x
            self.best_iter = self.curr_iter

        if self.curr_iter - self.best_iter == self.config.patience:
            LOGGER.info(f"Triggered Early Stopping at epoch {self.curr_iter + 1}")
            self.stop = True
        return self

    def cast(self):
        if self.mixed_precision:
            return torch.autocast(device_type=self.device.type, dtype=self.precision, enabled=True)
        return nullcontext
