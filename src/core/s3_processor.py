import io
import logging
from typing import Optional, List, Tuple
from google.cloud import storage  # TODO google.cloude.storage_v2.AsyncClient
import soundfile as sf
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import S3Config
    from google.cloud.storage import Bucket


logger = logging.getLogger(__name__)


class S3Processor:
    def __init__(self, s3_config: "S3Config"):
        self.config = s3_config
        self.client = storage.Client()
        self.input_bucket = self.client.bucket(s3_config.input_bucket_name)
        self.output_bucket = self.client.bucket(s3_config.output_bucket_name)

    def list_dir(self, bucket: "Bucket", path: str, extensions: Optional[List[str]] = None) -> list[str]:
        return [
            blob.name
            for blob in self.client.list_blobs(bucket, prefix=path)
            if any(blob.name.lower().endswith(ext) for ext in extensions or [])
        ]
    
    def list_input_audio_files(self, path: str = "", extensions: Optional[List[str]] = None) -> List[str]:

        if extensions is None:
            extensions = self.config.supported_extensions

        return self.list_dir(self.input_bucket, path, extensions)
    
    def list_results_jsons(self, path: str) -> list[str]:
        return self.list_dir(self.output_bucket, path, ['.json'])
    
    def download_audio(self, file_path) -> tuple[np.ndarray, int]:
        try:
            blob = self.input_bucket.blob(file_path)
            audio_bytes = blob.download_as_bytes()
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            logger.info(f"Downloaded: {file_path}")
            return audio_np, sr
        except Exception:
            logger.error(f"Error downloading {file_path}")
            raise

    def upload_result_json(self, file_path, content) -> None:
        try:
            self.output_bucket.blob(file_path).upload_from_string(content, content_type="application/json")
            logger.info(f"Uploaded: {file_path}")
        except Exception:
            logger.error(f"Error uploading {file_path}")
            raise
