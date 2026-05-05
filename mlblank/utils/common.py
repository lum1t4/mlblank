import logging.config
import os
from pathlib import Path
import platform

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # multiprocessing threads
VERBOSE = str(os.getenv("VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
LOGGING_NAME = ROOT.stem
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
PYTHON_VERSION = platform.python_version()


def set_logging(name=f"{__name__}", verbose: bool = True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different
    environments.
    """
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level,}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,}}})

    # Set up the logger
    logger = logging.getLogger(name)
    logger.propagate = False
    return logger


LOGGER = set_logging(name=LOGGING_NAME, verbose=True)  # logger