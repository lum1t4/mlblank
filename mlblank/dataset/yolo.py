import glob
import math
import os
from pathlib import Path

import cv2
import numpy as np
from mlblank.core.ops import segments2boxes

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]



def img2label(img_path):
    # Define label path as a function of image path
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return Path(sb.join(str(img_path).rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def list_imgs(path: str | Path, prefix="⚠️"):
    """Read image files."""
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                # F = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [
                        x.replace("./", parent) if x.startswith("./") else x for x in t
                    ]  # local to global path
                    # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f"{prefix}{p} does not exist")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert im_files, f"{prefix}No images found in {path}."
    except Exception as e:
        raise FileNotFoundError(f"{prefix}Error loading data from {path}\n") from e
    return im_files



def load_image(path: str | Path, imgsz: int = 640, augment: bool = False, stretch: bool = False, flag: int = cv2.IMREAD_UNCHANGED) ->  tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Load and preprocess an image from the given path.
    Args:
        path (str | Path): The path to the image file.
        imgsz (int, optional): The desired size of the image. Defaults to 640.
        augment (bool, optional): Whether to apply augmentation to the image. Defaults to False.
        rect (bool, optional): Whether to use rectangular resizing..
    Returns:
        A tuple containing the preprocessed image, the original height and width of the image, and the new height and width of the image.
    """
    path = Path(path)
    assert path.exists(), f"File not found: {path}"
    im = cv2.imread(path.as_posix(), flag)
    h0, w0 = im.shape[:2]  # orig hw
    if not stretch:  # resize long side to imgsz while maintaining aspect ratio
        r = imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
            interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (w, h), interpolation=interp)
    elif not (h0 == w0 == imgsz):  # resize by stretching image to square imgsz
        im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return im, (h0, w0), im.shape[:2]


def load_label(
    path: str | Path,
    num_classes: int | None = None,
    keypoint: bool = False,
    nkpt: int = 0,
    ndim: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses label files and returns a numpy array with class labels and bounding boxes."""

    labels, segments, keypoints = None, None, None

    path = Path(path) if isinstance(path, str) else path
    
    def empty_labels():
        # Return an empty array with 5 columns if no labels (or 5 + num_keypoint * num_dim if with keypoints)
        return np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
    
    if not path.exists() or path.stat().st_size == 0:
        return empty_labels()


    with open(path, "r", encoding="utf-8") as f:
        labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if any(len(x) > 6 for x in labels) and (not keypoint): # is segment
            classes = np.array([x[0] for x in labels], dtype=np.float32)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in labels]  # (cls, xy1...
            labels = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xyw

    labels = np.array(labels, dtype=np.float32)

    if nl := len(labels) <= 0:
        return empty_labels()

    if keypoint:
        assert labels.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
        points = labels[:, 5:].reshape(-1, ndim)[:, :2]
    else:
        assert labels.shape[1] == 5, f"labels require 5 columns, {labels.shape[1]} columns detected"
        points = labels[:, 1:]
     

    # Coordinate points check with 1% tolerance
    assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
    assert labels.min() >= -0.01, f"negative class labels or coordinate {labels[labels < -0.01]}"

    max_cls = labels[:, 0].max()  # max label class
    assert num_classes is None or max_cls < num_classes, f"Label class {int(max_cls)} exceeds dataset class count {num_classes}. Possible class labels are 0-{num_classes - 1}"

    _, index = np.unique(labels, axis=0, return_index=True)
    if len(index) < nl:  # Remove duplicates if any
        labels = labels[index]
        if segments:
            segments = [segments[x] for x in index]

    return labels, segments, keypoints
