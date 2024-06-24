from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomArguments:
    """
    Arguments to
    """
    target_sparsity: float = field(
        default=0.5, metadata={"help": "target_sparsity"}
    )
    window_size: int = field(
        default=3, metadata={"help": "Size of the Sliding-window"}
    )
    flooding_b: float = field(
        default=0.01, metadata={"help": "the level of flooding trick"}
    )
    retrain_learning_rate: float = field(
        default=2e-5, metadata={"help": "retrain learning rate"}
    )
    pruning_type: str = field(
        default=None, metadata={"help": "Type of pruning"}
    )
    prepruning_finetune_epochs: int = field(
        default=3, metadata={"help": "finetuning epochs"}
    )
    teacher_model_path: str = field(
        default=None, metadata={"help": "Path of the teacher model for distillation."}
    )
    do_finetune: bool = field(
        default=False, metadata={"help": "Whether to do finetune or not."}
    )
    do_distill: bool = field(
        default=False, metadata={"help": "Whether to do distillation or not."}
    )
    prune_epochs: int = field(
        default=10, metadata={"help": "Number of epochs for warmup"}
    )
