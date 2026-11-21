"""train's defaults: added where not given, and none when resuming."""

import unittest

from src.lerobot_policy.train import with_defaults


class DefaultsTest(unittest.TestCase):
    def test_added(self):
        args = with_defaults(["--dataset.root=recordings/grasp_lerobot", "--steps=10"])
        self.assertEqual(args, [
            "--policy.device=cuda", "--policy.push_to_hub=false", "--dataset.repo_id=local/grasp", "--job_name=act_grasp",
            "--policy.type=act", "--dataset.root=recordings/grasp_lerobot", "--steps=10",
        ])

    def test_given_ones_kept(self):
        args = with_defaults(["--dataset.root", "/data/cubes/", "--policy.type=diffusion", "--policy.push_to_hub=true",
                              "--dataset.repo_id=me/cubes", "--job_name", "mine"])
        self.assertEqual(args[0], "--policy.device=cuda")  # the only default not given
        self.assertEqual(len(args), 1 + 7)
        self.assertNotIn("--policy.type=act", args)

    def test_pretrained_policy_has_no_type(self):
        args = with_defaults(["--dataset.root=x/grasp_lerobot", "--policy.path=lerobot/act_aloha"])
        self.assertFalse(any(a.startswith("--policy.type") for a in args))
        self.assertIn("--job_name=pretrained_grasp", args)

    def test_resume_and_help_as_given(self):
        resume = ["--config_path=o/checkpoints/last/pretrained_model/train_config.json", "--resume=true"]
        self.assertEqual(with_defaults(resume), resume)
        self.assertEqual(with_defaults(["--help"]), ["--help"])

    def test_dataset_needed(self):
        with self.assertRaisesRegex(SystemExit, "--dataset.root"):
            with_defaults(["--steps=10"])


if __name__ == "__main__":
    unittest.main()
