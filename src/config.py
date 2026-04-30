"""Configuration du projet via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Hyperparamètres et configuration du projet."""

    # Chemins
    project_root: Path = Path(__file__).parent.parent
    data_raw: Path = project_root / "data" / "1-raw"
    data_processed: Path = project_root / "data" / "2-processed"
    data_external: Path = project_root / "data" / "3-external"
    outputs_models: Path = project_root / "outputs" / "models"
    outputs_logs: Path = project_root / "outputs" / "logs"
    outputs_results: Path = project_root / "outputs" / "results"

    # Entraînement
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    val_split: float = 0.2
    test_split: float = 0.1

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "default"

    # Modèle
    use_cuda: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
