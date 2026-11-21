"""PolicyRunner on the GPU: PyTorch finds CUDA (its libraries are deps: //src/lerobot_export:torch_cuda), and a
policy the size of a trained ACT, on the datasets' three cameras (720p, resized to the datasets' size), keeps up with 20 Hz. Needs an NVIDIA GPU."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.lerobot_export.conventions import CAMERAS
from src.lerobot_policy.policy_runner import PolicyRunner
from src.lerobot_policy.testing import CAMERA_IMAGE, DATASET_IMAGE, save_checkpoint

HOME = np.array([0.0, 1.2, 1.4, -0.7, 0.0, 0.0, 0.04])
PERIOD = 1.0 / 20  # the datasets' fps


class PolicyRunnerGpuTest(unittest.TestCase):
    def test_cuda(self):
        import torch

        self.assertTrue(torch.cuda.is_available(), f"torch {torch.__version__} (CUDA {torch.version.cuda}) finds no GPU")
        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}", flush=True)
        path = save_checkpoint(Path(tempfile.mkdtemp()) / "act", 7, cameras=tuple(CAMERAS), image=DATASET_IMAGE, small=False)
        with PolicyRunner(path, device="cuda", max_joint_step=10.0) as runner:
            self.assertEqual(next(runner.policy.parameters()).device.type, "cuda")
            rng = np.random.default_rng(0)
            images = {cam: rng.integers(0, 256, CAMERA_IMAGE, dtype=np.uint8) for cam in runner.cameras}  # resized by act
            runner.reset()
            chunk = runner.policy.config.n_action_steps
            times = []
            for _ in range(2 * chunk + 1):  # warm-up, then two chunks: a chunk's inference and the queued actions
                start = time.perf_counter()
                targets = runner.act(HOME, images)
                times.append(time.perf_counter() - start)
                self.assertEqual(targets.shape, (7,))
                self.assertTrue(np.all(np.isfinite(targets)))
            inference = times[chunk]  # the second chunk's first action: the policy ran
            queued = float(np.median(times[chunk + 1:]))
            print(f"chunk of {chunk}: {inference * 1e3:.1f} ms to infer, {queued * 1e3:.2f} ms per queued action", flush=True)
            self.assertLess(inference, PERIOD, "a chunk's inference does not fit in a 20 Hz period")


if __name__ == "__main__":
    unittest.main()
