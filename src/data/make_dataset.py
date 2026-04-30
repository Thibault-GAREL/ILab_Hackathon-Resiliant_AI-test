"""Chargement, nettoyage et split des données."""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_data(data_path: Path):
    """Charge les données brutes."""
    logger.info(f"Chargement des données depuis {data_path}")
    # À implémenter selon tes besoins
    pass


def preprocess_data(raw_data):
    """Nettoie et prépare les données."""
    logger.info("Nettoyage des données")
    # À implémenter selon tes besoins
    pass


def split_data(data, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split en train/val/test."""
    # À implémenter selon tes besoins
    pass
