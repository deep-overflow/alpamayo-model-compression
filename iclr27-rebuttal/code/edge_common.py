"""Shared pieces for the Alpamayo 2 Edge track (plans/2026-11-14_a2-edge-analysis-replication.md).

Runs in the Edge venv (/mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge: Python 3.12, torch 2.8.0, transformers 5.17.0),
not the repo .venv. Import this module before anything that touches the Hub: setup_hf_env() sets HF_TOKEN from
alpasim-runs/.hf_env (the Seongchan token) with HF_HOME / HF_HUB_CACHE, because a bare launch is anonymous and the
dataset SDK then gets 429s.

Prompt: Edge's own build_conversation with the navigation line removed (decision Q1 of the plan: the A2 Super runs had
no navigation line, the public dataset has no route). Edge's loader and validator require a nav_text entry, so a
placeholder is passed to them and asserted absent from the tokenized prompt.
"""

import hashlib
import io
import os
import re
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "nvidia/Alpamayo2-Edge"
MODEL_REV = "f8951d095de3589b3face1dec3f4394c8ecedb38"
AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
CALIB_MANIFEST = Path("/mnt/nvme1n1/ad_vla/outputs/chan/eval_sets/calib_100.parquet")
CALIB_CACHE = AV / "pre_processed" / "calib_edge"
NAV_PLACEHOLDER = "NAVIGATION_LINE_REMOVED"
NAV_PREFIX = "Navigation input is: "


def setup_hf_env(offline=False):
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/nvme1n1/ad_vla/cache/hub")
    if "HF_TOKEN" not in os.environ:
        txt = Path("/home/cvlab21/project/chan/alpasim-runs/.hf_env").read_text()
        os.environ["HF_TOKEN"] = re.search(r"HF_TOKEN=(\S+)", txt).group(1).strip("\"'")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"


def clip_seed(global_seed, clip_id):
    """sha256(f"{seed}:{clip_id}")[:4], the rule of every importance run (cache_a2.clip_seed)."""
    h = hashlib.sha256(f"{global_seed}:{clip_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def path_for(clip_id, t0_us, root=CALIB_CACHE):
    return Path(root) / "samples" / f"{clip_id}__t0_{t0_us}.npz"


def load_cached(path):
    """Rebuild the dict Edge's load_physical_aiavdataset returns, plus the GT future (ego frame at t0)."""
    from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
    from PIL import Image

    z = np.load(path, allow_pickle=True)
    n, H, W = len(z["jpeg_bytes"]), int(z["H"]), int(z["W"])
    frames = np.empty((n, 1, 3, H, W), dtype=np.uint8)  # (22, 1, 3, H, W)
    for i, b in enumerate(z["jpeg_bytes"]):
        frames[i, 0] = np.transpose(np.asarray(Image.open(io.BytesIO(b))), (2, 0, 1))
    return {
        "image_frames": torch.from_numpy(frames),
        "camera_indices": torch.from_numpy(z["camera_indices"]),  # (22,)
        "absolute_timestamps": torch.from_numpy(z["absolute_timestamps"]),  # (22,)
        "ego_history_xyz": torch.from_numpy(z["ego_history_xyz"]),  # (1, 20, 3)
        "ego_history_rot": torch.from_numpy(z["ego_history_rot"]),  # (1, 20, 3, 3)
        "ego_future_xyz": torch.from_numpy(z["ego_future_xyz"]),  # (1, 64, 3)
        "ego_future_rot": torch.from_numpy(z["ego_future_rot"]),  # (1, 64, 3, 3)
        "nav_text": [NAV_PLACEHOLDER],
        "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        "min_pixels": 163840,
        "max_pixels": 196608,
        "input_profile": "camera_subsampled",
        "history_num_steps": 20,
        "clip_id": str(z["clip_id"]),
        "t0_us": int(z["t0_us"]),
    }


def drop_nav_line(messages):
    """Remove the navigation item and the newline item that follows it from Edge's driving conversation."""
    content = messages[1]["content"]
    idx = [i for i, c in enumerate(content) if c["type"] == "text" and c["text"].startswith(NAV_PREFIX)]
    assert len(idx) == 1, f"expected one navigation item, found {len(idx)}"
    i = idx[0]
    assert content[i + 1] == {"type": "text", "text": "\n"}, content[i + 1]
    del content[i : i + 2]
    return messages


def prepare_inputs_no_nav(data, model_config, tokenizer, processor=None):
    """helper.prepare_model_inputs with the navigation line dropped. Same return layout."""
    from alpamayo2_edge import helper
    from alpamayo2_edge.chat_template import build_conversation

    canonical = helper._canonical_inputs(data, model_config)
    messages = build_conversation(
        canonical,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        include_camera_ids=model_config.include_camera_ids,
        system_prompt=getattr(model_config, "system_prompt", None),
    )
    messages = drop_nav_line(messages)
    processor = processor or helper.get_processor(tokenizer, model_config)
    tokenized_data = helper._tokenize_messages(messages, processor, include_closed_thinking=True)
    text = tokenizer.decode(tokenized_data["input_ids"][0])
    assert NAV_PLACEHOLDER not in text and "Navigation input" not in text
    xyz, rot = canonical["ego_history_xyz"], canonical["ego_history_rot"]
    return {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": xyz.unsqueeze(0) if xyz.ndim == 3 else xyz,  # (1, 1, 20, 3)
        "ego_history_rot": rot.unsqueeze(0) if rot.ndim == 4 else rot,  # (1, 1, 20, 3, 3)
    }, text


def load_model(device="cuda", attn="sdpa"):
    from alpamayo2_edge.checkpoint import load_model as _load

    model, info = _load(MODEL_ID, revision=MODEL_REV, device=device, attn_implementation=attn)
    return model, info
