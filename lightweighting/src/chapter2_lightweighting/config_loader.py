from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DistillConfig, ModelTopology, SearchBounds, SearchConfig, VehicleConstraint


@dataclass
class DataConfig:
    calibration_texts_path: str
    distill_texts_path: str
    max_length: int = 128
    batch_size: int = 2


@dataclass
class ModelConfig:
    base_model: str
    teacher_model: str
    reference_model: str
    tokenizer: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class OutputConfig:
    output_dir: str
    save_model_state: bool = True
    save_masks: bool = True
    save_report: bool = True


@dataclass
class RuntimeConfig:
    device: str = "auto"
    max_calibration_batches: int = 8
    seed: int = 42


@dataclass
class LightweightingRunConfig:
    model: ModelConfig
    data: DataConfig
    vehicle: VehicleConstraint
    search: SearchConfig
    distill: DistillConfig
    output: OutputConfig
    runtime: RuntimeConfig
    model_topology: Optional[ModelTopology] = None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(config_path: str, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(config_path).resolve().parent / path).resolve())


def _build_search_config(raw: Dict[str, Any]) -> SearchConfig:
    bounds_raw = raw.get("bounds", {})
    bounds = SearchBounds(**bounds_raw)
    payload = dict(raw)
    payload["bounds"] = bounds
    return SearchConfig(**payload)


def load_run_config(path: str) -> LightweightingRunConfig:
    raw = _load_json(path)

    data_raw = dict(raw["data"])
    data_raw["calibration_texts_path"] = _resolve_path(path, data_raw["calibration_texts_path"])
    data_raw["distill_texts_path"] = _resolve_path(path, data_raw["distill_texts_path"])

    output_raw = dict(raw["output"])
    output_raw["output_dir"] = _resolve_path(path, output_raw["output_dir"])

    topology = None
    if raw.get("model_topology"):
        topology = ModelTopology(**raw["model_topology"])

    return LightweightingRunConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**data_raw),
        vehicle=VehicleConstraint(**raw["vehicle"]),
        search=_build_search_config(raw["search"]),
        distill=DistillConfig(**raw["distill"]),
        output=OutputConfig(**output_raw),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        model_topology=topology,
    )

