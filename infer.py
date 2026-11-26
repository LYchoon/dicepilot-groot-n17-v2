"""Predict a DicePilot action chunk from one synchronized observation."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch
from gr00t.policy.gr00t_policy import Gr00tPolicy
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load_policy(variant="bc", base_dir=None, weights_dir=None):
    base_files = ["config.json", "model.safetensors.index.json", "model-*.safetensors"]
    base = Path(base_dir or snapshot_download(
        "nvidia/GR00T-N1.7-3B", revision="373d34e2723d928870157d07dc216a19a624b5bb",
        allow_patterns=base_files,
    )).resolve()
    patterns = ["bc-15000.safetensors", "processor/*"]
    if variant != "bc":
        patterns.append(f"{variant}.safetensors")
    weights = Path(weights_dir or snapshot_download(
        "LY0613/dicepilot-groot-n17-v2", allow_patterns=patterns,
    )).resolve()
    with TemporaryDirectory() as directory:
        model_dir = Path(directory)
        for pattern in base_files:
            for source in base.glob(pattern):
                (model_dir / source.name).symlink_to(source)
        (model_dir / "processor").symlink_to(weights / "processor", target_is_directory=True)
        policy = Gr00tPolicy("NEW_EMBODIMENT", str(model_dir), device="cuda:0")
    bc = load_file(str(weights / "bc-15000.safetensors"))
    if len(bc) != 613 or "action_head.action_decoder.layer2.W" not in bc:
        raise ValueError("Unexpected DicePilot BC checkpoint")
    result = policy.model.load_state_dict(bc, strict=False)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys)
    del bc
    if variant != "bc":
        head = load_file(str(weights / f"{variant}.safetensors"))
        policy.model.action_head.load_state_dict(head, strict=True)
    policy.model.eval().requires_grad_(False)
    policy.model.action_head.num_inference_timesteps = 4
    return policy


@torch.inference_mode()
def predict(policy, side, wrist, state, task):
    q = np.asarray(state, dtype=np.float32).reshape(1, 1, 6)
    obs = {
        "video": {"side": side[None, None], "wrist": wrist[None, None]},
        "state": {"single_arm": q[..., :5], "gripper": q[..., 5:]},
        "language": {policy.language_key: [[task]]},
    }
    actions, _ = policy.get_action(obs)
    chunk = np.concatenate([actions["single_arm"], actions["gripper"]], axis=-1)[0, :8]
    if chunk.shape != (8, 6) or not np.isfinite(chunk).all():
        raise ValueError("Invalid action chunk")
    return np.clip(chunk, [-100] * 5 + [0], 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["bc", "nft-best", "nft-last"], default="bc")
    parser.add_argument("--side", required=True)
    parser.add_argument("--wrist", required=True)
    parser.add_argument("--state", type=float, nargs=6, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-dir")
    parser.add_argument("--weights-dir")
    args = parser.parse_args()
    frames = []
    for path in (args.side, args.wrist):
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    policy = load_policy(args.variant, args.base_dir, args.weights_dir)
    print(json.dumps(predict(policy, *frames, args.state, args.task).tolist()))
