"""Compare learning rates in the assignment's decaying-SGD toy example."""

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor


class SGD(torch.optim.Optimizer):
    """SGD whose step size at iteration ``t`` is ``lr / sqrt(t + 1)``."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                iteration = state.get("iteration", 0)
                step_size = learning_rate / math.sqrt(iteration + 1)
                parameter.add_(parameter.grad, alpha=-step_size)
                state["iteration"] = iteration + 1

        return loss


def run_experiment(initial_weights: Tensor, learning_rate: float, iterations: int = 10) -> None:
    """Print the loss at every iteration for one learning rate."""

    weights = torch.nn.Parameter(initial_weights.clone())
    optimizer = SGD([weights], lr=learning_rate)

    print(f"\nlearning rate = {learning_rate:g}")
    for iteration in range(iterations):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        print(f"iteration {iteration + 1:2d}: loss = {loss.item():.8f}")
        loss.backward()
        optimizer.step()


def main() -> None:
    # Every learning rate starts from exactly the same randomly generated
    # weights, so learning rate is the only experimental variable.
    torch.manual_seed(0)
    initial_weights = 5 * torch.randn((10, 10))

    for learning_rate in (1e1, 1e2, 1e3):
        run_experiment(initial_weights, learning_rate, iterations=10)


if __name__ == "__main__":
    main()
