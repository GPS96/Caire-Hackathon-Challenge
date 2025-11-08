"""Application configuration settings."""

from functools import lru_cache
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    app_name: str = Field("HackHealthAI Driver Monitor", env="APP_NAME")
    app_version: str = Field("0.1.0", env="APP_VERSION")
    heartbeat_window_seconds: int = Field(15, env="HEARTBEAT_WINDOW")
    websocket_broadcast_interval: float = Field(1.0, env="WS_BROADCAST_INTERVAL")
    model_checkpoint_path: str = Field("ml/models/siamaf_pretrained.pt", env="MODEL_CHECKPOINT")
    device: str = Field("cpu", env="DEVICE")
    
    # Simulation mode
    simulation_mode: bool = Field(False, env="SIMULATION_MODE")
    sim_weights_dir: str = Field("data/arrhythmia_project/weights", env="SIM_WEIGHTS_DIR")
    sim_data_root: str = Field("data/training", env="SIM_DATA_ROOT")
    sim_sampling_rate: int = Field(125, env="SIM_SAMPLING_RATE")
    
    # Backend WebSocket Configuration
    api_key: str = Field("", env="API_KEY")
    backend_ws_base: str = Field("", env="BACKEND_WS_BASE")
    
    # Camera & Capture Settings
    fps: int = Field(30, env="FPS")
    camera_index: int = Field(0, env="CAMERA_INDEX")
    res_width: int = Field(640, env="RES_WIDTH")
    res_height: int = Field(480, env="RES_HEIGHT")
    
    # Frame Encoding
    frame_format: str = Field("jpeg", env="FRAME_FORMAT")
    jpeg_quality: int = Field(75, env="JPEG_QUALITY")
    
    # Optional WebSocket Parameters
    client: str = Field("livePython", env="CLIENT")
    object_id: str = Field("", env="OBJECT_ID")
    callback_url: str = Field("", env="CALLBACK_URL")
    
    # rPPG to PPG Conversion Settings
    rppg_buffer_sec: int = Field(20, env="RPPG_BUFFER_SEC")
    live_csv: str = Field("live_ppg.csv", env="LIVE_CSV")
    
    # UI Options
    show_preview: int = Field(1, env="SHOW_PREVIEW")
    show_plot: int = Field(1, env="SHOW_PLOT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
