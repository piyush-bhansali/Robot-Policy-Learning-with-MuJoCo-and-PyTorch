import argparse
import math

import torch
import wandb
from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.linspace(-2 * math.pi, 2 * math.pi, 1000).to(device).unsqueeze(1)
    y = torch.sin(x).to(device)

    model = MLP(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    run = wandb.init(project="physai", name=f"fit_sin_lr{args.lr}_h{args.hidden_dim}", config=vars(args) | {"device": device})

    for steps in range(args.num_epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if steps % 10 == 0:
            run.log({"loss": loss.item()}, step=steps)

        if steps % 200 == 0:
            print(f"step {steps}  loss {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        y_test = model(x)
        xs = x.squeeze(1).cpu().tolist()
        run.log({
            "fit": wandb.plot.line_series(
                xs=xs,
                ys=[y.squeeze(1).cpu().tolist(), y_test.squeeze(1).cpu().tolist()],
                keys=["ground truth", "predictions"],
                title="Fit",
                xname="x"
            )
        })
    run.finish()

if __name__ == "__main__":
    main()


