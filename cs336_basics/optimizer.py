"""Optimization algorithms implemented from first principles."""

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor


def cosine_learning_rate_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the warmup-and-cosine learning rate for one iteration."""

    if iteration < warmup_iters:
        return (iteration / warmup_iters) * max_learning_rate

    if iteration <= cosine_cycle_iters:
        cosine_progress = (iteration - warmup_iters) / (
            cosine_cycle_iters - warmup_iters
        )
        cosine_multiplier = 0.5 * (1 + math.cos(math.pi * cosine_progress))
        return min_learning_rate + cosine_multiplier * (
            max_learning_rate - min_learning_rate
        )

    return min_learning_rate


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay.

    Each parameter keeps its own step count and exponential moving averages of
    the gradient and squared gradient. The learning rate is bias-corrected
    before the moment-normalized update is applied.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Perform one AdamW update and optionally return the closure's loss."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                gradient = parameter.grad
                state = self.state[parameter]

                if not state:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]

                # Decoupled weight decay: theta <- theta - lr * lambda * theta.
                parameter.mul_(1 - learning_rate * weight_decay)

                first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                adjusted_learning_rate = (
                    learning_rate * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                )
                denominator = torch.sqrt(second_moment) + epsilon
                parameter.addcdiv_(
                    first_moment,
                    denominator,
                    value=-adjusted_learning_rate,
                )

        return loss
