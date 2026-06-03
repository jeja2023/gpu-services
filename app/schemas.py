import asyncio
from typing import Any, TypedDict

import onnxruntime as ort
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.model_refs import validate_path_name


class ModelBundle(TypedDict):
    session: ort.InferenceSession
    lock: asyncio.Lock
    semaphore: asyncio.Semaphore
    path: str
    model_hash: str
    file_size: int
    loaded_at: float
    last_used_at: float
    load_count: int
    inference_count: int
    max_concurrency: int
    queue_timeout_seconds: float


class LetterboxMeta(TypedDict):
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    pad_left: float
    pad_top: float


class ModelConfig(TypedDict, total=False):
    task: str
    type: str
    runtime: str
    version: str
    precision: str
    input_size: list[int]
    input: dict[str, Any]
    output: dict[str, Any]
    artifact: dict[str, Any]
    rollout: dict[str, Any]
    person_class_id: int
    confidence: float
    iou: float
    classes: str
    normalize: str
    embedding_normalize: str
    batch_size: int


class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_name: str = Field(..., min_length=1, max_length=128)
    model_name: str = Field(..., min_length=1, max_length=256)
    tensor_data: list[Any] = Field(..., min_length=1)

    @field_validator("project_name", "model_name")
    @classmethod
    def reject_path_segments(cls, value: str) -> str:
        return validate_path_name(value)


class ModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_name: str = Field(..., min_length=1, max_length=128)
    model_name: str = Field(..., min_length=1, max_length=256)

    @field_validator("project_name", "model_name")
    @classmethod
    def reject_path_segments(cls, value: str) -> str:
        return validate_path_name(value)


class WarmupRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    models: list[ModelRequest] = Field(..., min_length=1)


class AliasSwitchRequest(BaseModel):
    alias_name: str = Field(..., min_length=1, max_length=128)
    target_model_id: str = Field(..., min_length=3, max_length=384)
    expected_current_target: str | None = Field(default=None, min_length=3, max_length=384)
    dry_run: bool = False

    @field_validator("alias_name")
    @classmethod
    def reject_alias_path_segments(cls, value: str) -> str:
        return validate_path_name(value)

    @field_validator("target_model_id", "expected_current_target")
    @classmethod
    def reject_invalid_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parts = value.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("model id must use 'project_name/model_name' format")
        project, model = parts
        validate_path_name(project)
        validate_path_name(model)
        return value


class AliasRollbackRequest(BaseModel):
    alias_name: str = Field(..., min_length=1, max_length=128)
    dry_run: bool = False

    @field_validator("alias_name")
    @classmethod
    def reject_alias_path_segments(cls, value: str) -> str:
        return validate_path_name(value)


class AliasRolloutTarget(BaseModel):
    target_model_id: str = Field(..., min_length=3, max_length=384)
    weight: int = Field(..., ge=0, le=100_000)
    status: str | None = Field(default=None, max_length=64)

    @field_validator("target_model_id")
    @classmethod
    def reject_invalid_model_id(cls, value: str) -> str:
        parts = value.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("model id must use 'project_name/model_name' format")
        project, model = parts
        validate_path_name(project)
        validate_path_name(model)
        return value


class AliasWeightedRolloutRequest(BaseModel):
    alias_name: str = Field(..., min_length=1, max_length=128)
    targets: list[AliasRolloutTarget] = Field(..., min_length=1)
    expected_current_target: str | None = Field(default=None, min_length=3, max_length=384)
    dry_run: bool = False

    @field_validator("alias_name")
    @classmethod
    def reject_alias_path_segments(cls, value: str) -> str:
        return validate_path_name(value)

    @field_validator("expected_current_target")
    @classmethod
    def reject_invalid_expected_target(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parts = value.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("model id must use 'project_name/model_name' format")
        project, model = parts
        validate_path_name(project)
        validate_path_name(model)
        return value
