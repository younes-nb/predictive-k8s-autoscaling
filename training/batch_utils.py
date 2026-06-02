from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def find_max_batch_size(
    model,
    input_size,
    args,
    device,
    loss_fn: Optional[nn.Module] = None,
    starting_batch: int = 8192,
):
    batch_size = starting_batch
    model.train()

    while batch_size > 0:
        try:
            dummy_x = torch.randn(batch_size, args.input_len, input_size, device=device)
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


def find_max_inference_batch_size(
    model, input_size, args, device, starting_batch: int = 16384
):
    batch_size = starting_batch
    model.eval()

    while batch_size > 0:
        try:
            dummy_x = torch.randn(batch_size, args.input_len, input_size, device=device)
            with torch.no_grad():
                _ = model(dummy_x)

            del dummy_x
            torch.cuda.empty_cache()
            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise e

    raise RuntimeError("Could not find a batch size that fits in memory.")
