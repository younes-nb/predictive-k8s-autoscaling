"""Model registry: versioned artifacts, atomic writes, rollback.

Layout under ``registry_dir``::

    registry/
      index.json                 # {current: <version>, versions: [...]}
      <version>/
        model.pt                 # state_dict + metadata
        feature_schema.json
        validation.json
        README.md

A write goes to a temp directory then is renamed, so a half-written version can
never become ``current``. ``rollback`` just repoints ``current`` at the previous
version; nothing is deleted.
"""

import json
import os
import shutil
import tempfile
import time

import torch


class ModelRegistry:
    def __init__(self, root):
        self.root = root
        self.index_path = os.path.join(root, "index.json")
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(self.index_path):
            self._write_index({"current": None, "versions": []})

    # ------------------------------------------------------------------
    def _read_index(self):
        with open(self.index_path) as fh:
            return json.load(fh)

    def _write_index(self, idx):
        d = os.path.dirname(self.index_path)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(idx, fh, indent=2)
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------------
    def publish(self, model, version, feature_schema, validation, notes="",
                norm=None):
        idx = self._read_index()
        if version in idx["versions"]:
            raise ValueError(f"version {version} already published")
        parent = os.path.join(self.root, version)
        os.makedirs(parent, exist_ok=False)
        payload = {
            "state_dict": model.state_dict(),
            "classes": list(model.bottleneck.classes),
            "temperature": float(model.bottleneck.temperature),
            "dist_threshold": model.bottleneck._dist_threshold,
            "n_features": int(model.encoder.proj.in_features),
            "d_model": int(model.encoder.d_model),
            "norm": norm,
        }
        tmp = parent + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        try:
            torch.save(payload, os.path.join(tmp, "model.pt"))
            with open(os.path.join(tmp, "feature_schema.json"), "w") as fh:
                json.dump(feature_schema, fh, indent=2)
            with open(os.path.join(tmp, "validation.json"), "w") as fh:
                json.dump(validation, fh, indent=2)
            with open(os.path.join(tmp, "README.md"), "w") as fh:
                fh.write(f"# AURORA-HPA model `{version}`\n\n")
                fh.write(f"published: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
                fh.write(f"notes: {notes}\n\n")
                fh.write("## validation\n\n```json\n")
                fh.write(json.dumps(validation, indent=2))
                fh.write("\n```\n")
            os.replace(tmp, parent)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        idx["versions"].append(version)
        idx["current"] = version
        self._write_index(idx)
        return version

    # ------------------------------------------------------------------
    def load(self, version=None, registry_cls=None):
        idx = self._read_index()
        version = version or idx.get("current")
        if not version:
            raise FileNotFoundError("no published model")
        path = os.path.join(self.root, version, "model.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = {"version": version,
                "classes": payload.get("classes"),
                "temperature": payload.get("temperature"),
                "dist_threshold": payload.get("dist_threshold"),
                "n_features": payload.get("n_features"),
                "d_model": payload.get("d_model"),
                "norm": payload.get("norm")}
        return payload, meta

    # ------------------------------------------------------------------
    def rollback(self):
        idx = self._read_index()
        if len(idx["versions"]) < 2:
            raise RuntimeError("no previous version to roll back to")
        i = idx["versions"].index(idx["current"])
        if i == 0:
            raise RuntimeError("already at the oldest version")
        idx["current"] = idx["versions"][i - 1]
        self._write_index(idx)
        return idx["current"]

    def history(self):
        return self._read_index()