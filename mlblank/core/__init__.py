from pydantic import BaseModel


class BaseModelConfig(BaseModel):
    id2label: dict[int, str] | None = None
    label2id: dict[str, int] | None = None
    num_labels: int | None = None


class BaseTrainConfig(BaseModel):
    workers: int = 8
    seed: int = 0
    deterministic: bool = False
    batch_size: int = 4
    epochs: int = 10
    patience: int = 2
    save: bool = False
    save_period: int = -1
    device: str = "cpu"
