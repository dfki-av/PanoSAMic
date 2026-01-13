"""
Author: Mahdi Chamseddine
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class PanoSAMicLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        start_lr: float,
        max_lr: float,
        warm_up_steps: int,
        wind_down_step: int,
        total_steps: int,
        intermediate_lr: float | None = None,
    ):
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.warm_up_steps = warm_up_steps
        self.wind_down_step = wind_down_step
        self.total_steps = total_steps
        self.intermediate_lr = (
            intermediate_lr if intermediate_lr is not None else max_lr
        )  # Default to constant max_lr

        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int):
        """Computes the learning rate factor based on the current step."""
        match [step < self.warm_up_steps, step < self.wind_down_step]:
            case [True, True]:
                # Linear warm-up
                return self.start_lr + (
                    (self.max_lr - self.start_lr) * (step / self.warm_up_steps)
                )
            case [False, True]:
                if self.intermediate_lr == self.max_lr:
                    # Fixed value
                    return self.max_lr
                else:
                    # Linear decay to intermediate_lr
                    return self.max_lr - (step - self.warm_up_steps) * (
                        self.max_lr - self.intermediate_lr
                    ) / (self.wind_down_step - self.warm_up_steps)
            case [False, False]:
                # Linear decay to zero
                return self.intermediate_lr * (
                    1
                    - (step - self.wind_down_step)
                    / (self.total_steps - self.wind_down_step)
                )
            case _:
                raise ValueError
