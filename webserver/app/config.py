from pydantic.v1 import BaseSettings
import json
from typing import Optional

class GCPSettings(BaseSettings):
    project_id: str
    zone: str
    instance_name: str

class Settings(BaseSettings):
    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path) as f:
            config = json.load(f)
        return cls(**config)
    
    allowed_bucket: str = "default-bucket"
    redis_host: str = "localhost"
    redis_port: int = 6379
    gcp_settings: Optional[GCPSettings] = None
    shutdown_enabled: bool = False

try:
    settings = Settings.from_json('config.json')
except FileNotFoundError:
    settings = Settings()