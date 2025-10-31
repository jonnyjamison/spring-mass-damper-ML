from __future__ import annotations
import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from ..logging_conf import configure_logging
from ..utils.io import ensure_dir
from ..data.loader import load_config, build_training_data, DataConfig
from ..models.surrogate import MLP


def train(config_path: str):
    configure_logging()

    cfg = load_config(config_path)
    exp = cfg["experiment"]
    data_cfg = DataConfig(**cfg["data"])
    model_cfg = cfg["model"]
    art_cfg = cfg["artifacts"]
    mlflow_cfg = cfg.get("mlflow", {"enabled": False})

    if mlflow_cfg.get("enabled"):
        import mlflow
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(exp["name"])
        mlflow.start_run()
        mlflow.log_params({**exp, **model_cfg})

    X, Y = build_training_data(data_cfg)

    torch.manual_seed(exp["seed"]) # reproducibility

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=model_cfg["batch_size"], shuffle=True)


    model = MLP(3, model_cfg["hidden_sizes"], Y.shape[1])
    opt = optim.Adam(model.parameters(), lr=float(model_cfg["lr"]))
    loss_fn = nn.MSELoss()


    for epoch in range(int(model_cfg["epochs"])):
        model.train()
        running = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / len(dataset)
        if mlflow_cfg.get("enabled"):
            import mlflow
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: loss={epoch_loss:.6f}")

    # Save artifact
    out_dir = ensure_dir(art_cfg["output_dir"])
    out_path = Path(out_dir) / art_cfg["model_filename"]
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")

    if mlflow_cfg.get("enabled"):
        import mlflow
        mlflow.log_artifact(str(out_path))
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)