from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import math
from mlblank.dataset.yolo import img2label, load_image, load_label
import torch
from torch.utils.data import Dataset
import cv2


def render_heatmap_from_bboxes(
        bboxes: np.ndarray,
        image_width: int,
        image_height: int,
        mask_size: int,
        sigma: float,
        magnitude: float,
    ):
        x = bboxes[:, 1:]
        r = mask_size / max(image_height, image_width)
        w, h = (min(math.ceil(image_width * r), mask_size), min(math.ceil(image_height * r), mask_size))
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if x.size > 0:
            y = np.empty_like(x)
            y[..., 0] = x[..., 0] * w  # cx
            y[..., 1] = x[..., 1] * h  # cy
            y[..., 2] = x[..., 2] * w  # w
            y[..., 3] = x[..., 3] * h  # h
            gx, gy = np.meshgrid(np.arange(w), np.arange(h))
            for bx, by, bw, bh in y:
                s = max(1.0, min(sigma, bw / 2, bh / 2))
                g = np.exp(-(((gx - bx)**2 + (gy - by)**2) / (2 * s**2)))
                heatmap += g * magnitude
        
        return heatmap


def render_heatmap_from_segments(
    segments: list[np.ndarray],
    image_width: int,
    image_height: int,
    mask_size: int,
) -> np.ndarray:
    """
    Render a soft heatmap from segmentation polygons.

    Each segment is expected to be Nx2 with normalized coordinates in [0, 1].
    The polygon is rasterized on the target mask, then softly blurred.
    """
    r = mask_size / max(image_height, image_width)
    w = min(math.ceil(image_width * r), mask_size)
    h = min(math.ceil(image_height * r), mask_size)

    heatmap = np.zeros((h, w), dtype=np.float32)

    if not segments:
        return heatmap

    for seg in segments:
        if seg is None or len(seg) < 3:
            continue

        # normalized xy -> mask pixel coordinates
        poly = seg.copy().astype(np.float32)
        poly[:, 0] *= w
        poly[:, 1] *= h
        poly = np.round(poly).astype(np.int32)

        # clip to valid image range
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

        object = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(object, [poly], 1.0)
        heatmap += object

    return heatmap



class RacketDataset(Dataset):
    def __init__(
        self,
        index: Path | str,
        dataset: Path | str,
        data: dict = {"names": {0: "sports ball", 1: "person"}},
        classes: list[int] = [0],
        split: str = "train",
        imgsz: int = 640,
        masksz: int = 640,
        augment: int = False,
        sigma: float = 2.5,
        magnitude: float = 1.0,
        modality: Literal['detect', 'motion', 'sequence'] = 'detect',
        seq_length: int = 3,
        seq_stride: int = 3
    ):
        self.dataset = Path(dataset)
        self.split = split
        self.imgsz = imgsz
        self.masksz = masksz
        self.augment = augment
        self.names = data["names"]
        self.classes = classes
        self.num_classes = len(self.classes)
        self.sigma = sigma
        self.magnitude = magnitude
        self.modality = modality
        self.seq_length = seq_length
        self.seq_stride = seq_stride

        assert self.dataset.exists(), f"Dataset not fount at {self.dataset.as_posix()}"
        assert self.num_classes > 0
        df = pd.read_csv(index)
        df = df[df["split"] == split]
        df = df[df["sport"] != 'shuttlecock']
        self.im_files = list(map(lambda x: self.dataset / x, df["images"]))
        self.lb_files = list(map(img2label, self.im_files))

        if self.modality == 'motion':
            self.seq_length = 5
            self.seq_stride = 3

        if self.modality in {'motion', 'sequence'}:
            df['clip'] = list(map(lambda x: x.parent.parent, self.im_files))
            df = df.sort_values(['clip', 'images']).reset_index(drop=True)
            self.clips = df.groupby('clip', sort=False).groups
            self.indices = []
            for clip_name, clip_items in self.clips.items():
                for i in range(0, len(clip_items) - self.seq_length + 1, self.seq_stride):
                    self.indices.append(clip_items[i:i + self.seq_length])

    def __len__(self) -> int:
        return len(self.lb_files if self.modality == 'detect' else self.indices)
    

    def get_image_and_label(self, index: int) ->dict:
        im_file = self.im_files[index]
        lb_file = self.lb_files[index]

        image, original_shape, resized_shape \
            = load_image(im_file, imgsz=self.imgsz, augment=self.augment)
        
        # lb has the following schema: [cls, x, y, w, h]
        lb, segments, _ = load_label(lb_file)

        # filter labels according to classes
        filter_index = np.isin(lb[:, 0], self.classes)
        lb = lb[filter_index]
        segments = [segment for segment, x in zip(segments, filter_index) if x]
        
        # heatmap = self.render_heatmap_from_bboxes(lb, w_new, h_new, self.masksz, self.sigma, self.magnitude)
        heatmap = render_heatmap_from_segments(
            segments=segments,
            image_width=resized_shape[1],
            image_height=resized_shape[0],
            mask_size=self.masksz,
        )

        return {
            "img": image, "path": im_file, "original_shape": original_shape, "resized_shape": resized_shape,
            "format": "xywhn", "cls": lb[:, 0:1], "bboxes": lb[:, 1:], "normalize": True,
            "target": heatmap,
        }
    
    def get(self, index: int) -> dict:
        item = self.get_image_and_label(index)
        # BGR -> RGB
        img = item['img'][..., ::-1]
        # (H, W, C) -> (C, H, W)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        item['img'] = torch.from_numpy(img)
        item['target'] = torch.from_numpy(item['target']).unsqueeze(0)
        item["cls"] = torch.from_numpy(item["cls"])
        item["bboxes"] = torch.from_numpy(item["bboxes"])
        return item

    def __getitem__(self, index: int):
        if self.modality == 'sequence':
            new_item = {}
            xs, ys = [], []
            for idx in self.indices[index]:
                item = self.get(idx)
                xs.append(item['img'])
                ys.append(item['target'])

            new_item['img'] = torch.concat(xs)
            new_item['target'] = torch.concat(ys)
            return new_item
        
        if self.modality == 'motion':
            x_idx, y_idx = self.indices[index][0], self.indices[index][-1]
            x = self.get(x_idx)
            y = self.get(y_idx)
            # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
            weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            x_gray = (x['img'] * weights).sum(dim=0, keepdim=True)
            y_gray = (y['img'] * weights).sum(dim=0, keepdim=True)
            motion = torch.abs(x_gray - y_gray)  # (1, H, W)
            x['img'] = torch.concat((x['img'], motion))  # (4, H, W)
            return x

        return self.get(index)
        
    
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, key in enumerate(keys):
            value = values[i]
            if key in {"img", "target"}:
                value = torch.stack(value, 0)
            new_batch[key] = value
        return new_batch

