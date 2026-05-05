from datetime import datetime
import glob
import os
from pathlib import Path
import re
from typing import Optional

import yaml


def yaml_save(file: str = "data.yaml", data: Optional[dict] = None, header: str = ""):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file: str = "data.yaml", append_filename: bool = False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["__path__"] = str(file.absolute())
        return data



def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def file_age(path=__file__):
    """Return days since last file update."""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    """Return human-readable file modification date, i.e. '2021-3-26'."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path):
    """Return file/dir size (MB)."""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0


def get_latest_run(search_dir="."):
    """Return path to most recent 'last.pt' in /runs (i.e. to --resume from)."""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""

