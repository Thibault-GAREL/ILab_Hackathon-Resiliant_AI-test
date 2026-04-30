"""Boucle d'entraînement + MLflow tracking."""

import logging

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, config):
    """Entraîne une époque."""
    logger.info("Démarrage de l'entraînement")
    # À implémenter selon tes besoins
    pass


def train(model, train_loader, val_loader, config, num_epochs: int = 10):
    """Boucle d'entraînement principale avec MLflow tracking."""
    logger.info(f"Entraînement pour {num_epochs} épochs")
    # À implémenter selon tes besoins
    pass


def validate(model, val_loader):
    """Valide le modèle."""
    logger.info("Validation en cours")
    # À implémenter selon tes besoins
    pass
