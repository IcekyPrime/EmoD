from .base import BaseTrainer
from .stage1 import Stage1Trainer
from .stage2 import Stage2Trainer
from .stage3 import Stage3Trainer

_TRAINERS = {1: Stage1Trainer, 2: Stage2Trainer, 3: Stage3Trainer}


def build_trainer(cfg, dry_run: bool = False) -> BaseTrainer:
    stage = int(cfg["stage"])
    if stage not in _TRAINERS:
        raise ValueError(f"unknown stage {stage}; expected 1, 2 or 3")
    return _TRAINERS[stage](cfg, dry_run=dry_run)


__all__ = ["BaseTrainer", "Stage1Trainer", "Stage2Trainer", "Stage3Trainer", "build_trainer"]
