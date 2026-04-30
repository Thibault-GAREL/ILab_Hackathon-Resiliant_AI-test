"""Fonctions réutilisables partout dans le projet."""

import logging
from pathlib import Path


def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Configure un logger pour le projet."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dirs_exist(*paths: Path) -> None:
    """Crée les dossiers s'ils n'existent pas."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
