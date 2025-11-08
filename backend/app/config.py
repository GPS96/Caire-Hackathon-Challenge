"""Application configuration settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field("HackHealthAI Driver Monitor", env="APP_NAME")
    app_version: str = Field("0.1.0", env="APP_VERSION")
    heartbeat_window_seconds: int = Field(15, env="HEARTBEAT_WINDOW")
    websocket_broadcast_interval: float = Field(1.0, env="WS_BROADCAST_INTERVAL")
    model_checkpoint_path: str = Field("ml/models/siamaf_pretrained.pt", env="MODEL_CHECKPOINT")
    device: str = Field("cuda", env="DEVICE")
    # Simulation mode: ignore real rPPG and use dataset-backed predictions
    simulation_mode: bool = Field(True, env="SIMULATION_MODE")
    sim_weights_dir: str = Field("data/arrhythmia_project/weights", env="SIM_WEIGHTS_DIR")
    sim_data_root: str = Field("data/training", env="SIM_DATA_ROOT")
    sim_sampling_rate: int = Field(125, env="SIM_SAMPLING_RATE")
    # Simulation sampling controls
    sim_split: str = Field("all", env="SIM_SPLIT")  # one of: all, train, test
    sim_normal_ratio: float = Field(0.8, env="SIM_NORMAL_RATIO")  # desired probability of 'Normal' windows
    sim_flip_probability: float = Field(0.25, env="SIM_FLIP_PROB")  # sometimes invert ratio to 20/80
    sim_seed: int = Field(42, env="SIM_SEED")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
