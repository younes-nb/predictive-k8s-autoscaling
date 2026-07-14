import os
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Subset

from shared.config_paths import PATHS


def find_max_batch_size(
    model,
    input_size,
    args,
    device,
    loss_fn: Optional[nn.Module] = None,
    starting_batch=32768,
):
    batch_size = starting_batch
    model.train()

    while batch_size > 0:
        try:
            num_targets = getattr(model, 'num_targets', 1)
            dummy_x = torch.randn(batch_size, args.input_len, input_size, device=device)
            if num_targets > 1:
                dummy_y = torch.randn(batch_size, args.pred_horizon, num_targets, device=device)
            else:
                dummy_y = torch.randn(batch_size, args.pred_horizon, device=device)

            optimizer_dummy = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer_dummy.zero_grad()

            preds = model(dummy_x)
            if loss_fn is None:
                loss = ((preds - dummy_y) ** 2).mean()
            else:
                loss = loss_fn(preds, dummy_y)
            loss.backward()

            optimizer_dummy.zero_grad()
            del dummy_x, dummy_y, preds, loss
            torch.cuda.empty_cache()

            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise e

    raise RuntimeError("Could not find a batch size that fits in memory.")


def load_resume_state(path):
    if not os.path.exists(path):
        return None
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        logging.warning("Failed to load resume state: %s", exc)
        return None


def save_resume_state(path, state):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def head_slice_dataset_by_pct(dataset, pct: float):
    total = len(dataset)
    pct = float(pct)
    if pct <= 0 or pct >= 100 or total == 0:
        return dataset
    max_samples = max(1, int(total * pct / 100.0))
    if max_samples >= total:
        return dataset
    return Subset(dataset, range(max_samples))
