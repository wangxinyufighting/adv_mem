import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from training.dataset_builder import DatasetFiles


@dataclass(frozen=True)
class VerlConfig:
    epochs: int = 1
    batch_size: int = 8
    n_gpus: int = 1
    max_prompt_length: int = 4096


class VerlRunner:
    """Launch verl GRPO and convert its latest FSDP actor to Hugging Face format."""

    def __init__(self, root: str | Path, config: VerlConfig | None = None):
        self.root = Path(root)
        self.verl_dir = Path(
            os.environ.get("VERL_HOME", self.root / ".verl-cu124-src")
        )
        self.python = os.environ.get(
            "PYTHON_BIN", str(self.root / ".venv-cu124/bin/python")
        )
        self.config = config or VerlConfig()

    def train(
        self,
        role: str,
        model_path: str,
        dataset: DatasetFiles,
        output_dir: str | Path,
    ) -> str:
        # Verl runs from its own source directory, so paths crossing that process
        # boundary must be absolute.
        output = Path(output_dir).resolve()
        checkpoints = output / "checkpoints"
        merged = output / "model"
        batch_size = min(self.config.batch_size, dataset.train_size)
        env = {
            **os.environ,
            "MODEL_PATH": model_path,
            "TRAIN_FILE": str(dataset.train.resolve()),
            "VAL_FILE": str(dataset.val.resolve()),
            "NGPUS": str(self.config.n_gpus),
            "TRAIN_BATCH_SIZE": str(batch_size),
            "PPO_MINI_BATCH_SIZE": str(batch_size),
            "TOTAL_EPOCHS": str(self.config.epochs),
            "CHECKPOINT_DIR": str(checkpoints),
            "MAX_PROMPT_LENGTH": str(self.config.max_prompt_length),
        }
        if role == "attacker":
            env["ATTACKER_REWARD_TRACE"] = str(
                checkpoints / "rollouts" / "reward_trace.jsonl"
            )
        script = self.root / "scripts" / f"train_{role}_grpo.sh"
        subprocess.run([str(script)], cwd=self.root, env=env, check=True)

        actor = self._latest_checkpoint(checkpoints) / "actor"
        subprocess.run(
            [
                self.python,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor),
                "--target_dir",
                str(merged),
            ],
            cwd=self.verl_dir,
            check=True,
        )
        return str(merged)

    @staticmethod
    def _latest_checkpoint(directory: Path) -> Path:
        checkpoints = list(directory.glob("global_step_*"))
        if not checkpoints:
            raise RuntimeError(f"verl produced no checkpoint in {directory}")
        return max(checkpoints, key=lambda path: int(path.name.rsplit("_", 1)[1]))
