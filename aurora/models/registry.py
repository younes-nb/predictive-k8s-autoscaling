"""Model registry for the two-model AURORA bundle.

Layout under ``registry_dir``::

    registry/
      index.json                 # {current: <version>, versions: [...]}
      <version>/
        bundle.pt                # diagnosis + replica + schemas + OOD calibration
        validation.json
        README.md

A write goes to a temp directory and is renamed, so a half-written version can
never become ``current``. ``rollback`` repoints ``current`` at the previous
version; nothing is deleted.

The bundle stores both models plus the *exact* feature schema (order, units,
edge order, label taxonomy, normalization) and the OOD calibration, so a
checkpoint cannot load against a semantically different transform. The previous
registry assumed a single ``model.bottleneck`` / ``model.encoder`` object; that
interface no longer exists.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time

import torch


def feature_schema_hash(feature_order, edge_order):
    blob = json.dumps({"node": list(feature_order), "edge": list(edge_order)},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class BundleRegistry:
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
    def publish_candidate(self, bundle, version, validation, notes=""):
        """Write ``bundle`` as ``version`` without promoting it.

        ``bundle`` is the dict produced by ``train/train_models.py``. It must
        already carry the schema and OOD fields.
        """
        idx = self._read_index()
        if version in idx["versions"]:
            raise ValueError(f"version {version} already published")
        parent = os.path.join(self.root, version)
        os.makedirs(parent, exist_ok=False)
        tmp = parent + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        try:
            torch.save(bundle, os.path.join(tmp, "bundle.pt"))
            with open(os.path.join(tmp, "validation.json"), "w") as fh:
                json.dump(validation, fh, indent=2)
            with open(os.path.join(tmp, "README.md"), "w") as fh:
                fh.write(f"# AURORA bundle `{version}`\n\n")
                fh.write(f"published: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
                fh.write(f"notes: {notes}\n\n")
                fh.write(f"feature_schema_hash: "
                         f"{bundle.get('feature_schema_hash')}\n\n")
                fh.write("## validation\n\n```json\n")
                fh.write(json.dumps(validation, indent=2))
                fh.write("\n```\n")
            os.replace(tmp, parent)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        idx["versions"].append(version)
        # Publication does NOT promote. Promotion is a separate, explicit step
        # so a candidate can be validated or shadowed first.
        self._write_index(idx)
        return version

    def publish(self, bundle, version, validation, notes=""):
        """Publish and promote in one step (used by the training CLI)."""
        self.publish_candidate(bundle, version, validation, notes=notes)
        return self.promote(version)

    def promote(self, version):
        idx = self._read_index()
        if version not in idx["versions"]:
            raise KeyError(f"version {version} is not published")
        idx["current"] = version
        self._write_index(idx)
        return version

    # ------------------------------------------------------------------
    def load(self, version=None):
        """Load the published bundle and its metadata."""
        idx = self._read_index()
        version = version or idx.get("current")
        if not version:
            raise FileNotFoundError("no published model")
        path = os.path.join(self.root, version, "bundle.pt")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        meta = {"version": version,
                "feature_schema_hash": bundle.get("feature_schema_hash"),
                "n_node_features": bundle.get("n_node_features"),
                "d_model": bundle.get("d_model"),
                "n_degradation": bundle.get("n_degradation"),
                "n_edge_features": bundle.get("n_edge_features")}
        return bundle, meta

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