from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    media_root: Path = Path(os.environ.get("MEDIA_ROOT", "/mnt/user/data"))
    output_root: Path = Path(os.environ.get("OUTPUT_ROOT", "/output"))
    data_root: Path = Path(os.environ.get("DATA_ROOT", "/data"))
    model_path: Path = Path(
        os.environ.get("MODEL_PATH", "/models/2x-StarSample-V2-Lite.safetensors")
    )
    tile: int = int(os.environ.get("UPSCALE_TILE", "256"))
    context: int = int(os.environ.get("UPSCALE_CONTEXT", "32"))

    @property
    def database_path(self) -> Path:
        return self.data_root / "jobs.sqlite3"

    @property
    def log_root(self) -> Path:
        return self.data_root / "logs"


settings = Settings()
