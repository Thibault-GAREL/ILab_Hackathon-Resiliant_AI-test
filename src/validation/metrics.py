"""Calcul des métriques + MLflow artifacts."""

import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred):
    """Calcule les métriques du modèle."""
    logger.info("Calcul des métriques")
    # À implémenter selon tes besoins (accuracy, F1, etc.)
    pass


def log_artifacts(metrics: dict, output_path):
    """Enregistre les artefacts via MLflow."""
    logger.info(f"Enregistrement des artefacts dans {output_path}")
    # À implémenter selon tes besoins
    pass


def plot_metrics(metrics: dict, output_path):
    """Génère les graphiques de métriques."""
    logger.info(f"Génération des graphiques dans {output_path}")
    # À implémenter selon tes besoins
    pass
