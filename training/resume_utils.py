from __future__ import annotations

import logging
import os

import torch


def load_resume_state(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        logging.warning("Failed to load resume state: %s", exc)
        return None


def save_resume_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
