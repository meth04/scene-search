from pathlib import Path

from pydantic import BaseModel

from src.indexing.config import PipelineConfig, load_pipeline_config, load_structured
from src.indexing.domain.entities import DatasetManifest


class RunIndexCommand(BaseModel):
    manifest: DatasetManifest
    config: PipelineConfig

    @classmethod
    def from_files(cls, manifest_path: str | Path, config_path: str | Path):
        return cls(
            manifest=DatasetManifest.model_validate(load_structured(manifest_path)),
            config=load_pipeline_config(config_path),
        )
