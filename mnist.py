import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  

class MLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, test_loader

@hydra.main(version_base=None, config_path="configs", config_name="mnist")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, log_with="wandb")
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={"wandb": {"name": cfg.wandb.name}},
    )
    accelerator.print(OmegaConf.to_yaml(cfg))

    # With several GPUs, only one process downloads MNIST; the others wait
    with accelerator.main_process_first():
        train_loader, test_loader = get_dataloaders(cfg)

    model = MLP(cfg.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Moves model to GPU and makes loaders return batches already on the GPU
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    step = 0
    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            accelerator.backward(loss)
            optimizer.step()

            # loss is the average over this batch, so x batch size = batch total
            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)

            if step % cfg.log_every == 0:
                accelerator.log({"train/loss": loss.item(), "epoch": epoch}, step=step)
            step += 1

        train_loss = train_loss_sum / train_count

        # ---- evaluate on the 10,000 test images ----
        model.eval()
        correct, total, test_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images)
                logits, labels = accelerator.gather_for_metrics((logits, labels))
                test_loss_sum += loss_fn(logits, labels).item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        acc = correct / total
        test_loss = test_loss_sum / total

        accelerator.log(
            {"train/epoch_loss": train_loss, "test/loss": test_loss, "test/accuracy": acc},
            step=step,
        )
        accelerator.print(
            f"epoch {epoch}  train loss {train_loss:.4f}  test loss {test_loss:.4f}  test acc {acc:.4f}"
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()

    
