"""Single source of truth for shapes, paths and training knobs.

I keep this as a plain dataclass on purpose. A YAML file for six values
is how projects end up with hydra for no reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # data
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    split_file: Path = PROJECT_ROOT / "data" / "splits.csv"
    img_size: Tuple[int, int] = (160, 160)
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    # training
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 3e-4
    early_stop_patience: int = 6

    # outputs
    model_dir: Path = PROJECT_ROOT / "outputs" / "models"
    results_dir: Path = PROJECT_ROOT / "outputs" / "results"
    log_dir: Path = PROJECT_ROOT / "outputs" / "logs"

    CLASSES: Tuple[str, str] = field(default=("Normal", "Tuberculosis"))

    @property
    def model_path(self) -> Path:
        return self.model_dir / "tb_cnn.keras"

    def save(self, path: Path) -> None:
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        path.write_text(json.dumps(payload, indent=2))
