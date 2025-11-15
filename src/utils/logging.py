import logging
from typing import Optional


def get_logger(name: str = "wheat") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        logger.propagate = False
    return logger

