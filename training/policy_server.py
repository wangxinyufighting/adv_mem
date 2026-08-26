import os
import signal
import subprocess
import time
from contextlib import AbstractContextManager
from pathlib import Path
from urllib.request import urlopen

from openai import OpenAI


class ChatPolicy:
    def __init__(self, base_url: str, model: str):
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)

    def generate(self, prompt: list[dict[str, str]], max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=0.7,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (response.choices[0].message.content or "").strip()


class VLLMPolicyServer(AbstractContextManager[ChatPolicy]):
    """Serve one merged policy checkpoint during the inference half of a round."""

    def __init__(
        self,
        verl_dir: str | Path,
        model_path: str | Path,
        port: int,
        log_path: str | Path,
        gpu_memory_utilization: float = 0.8,
    ):
        self.verl_dir = Path(verl_dir)
        self.model_path = str(model_path)
        self.port = port
        self.log_path = Path(log_path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def __enter__(self) -> ChatPolicy:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        command = [
            os.environ.get("PYTHON_BIN", "python3"),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--served-model-name",
            "policy",
            "--dtype",
            "bfloat16",
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.verl_dir,
            env=os.environ.copy(),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self._wait_until_ready()
        except Exception:
            self.__exit__()
            raise
        return ChatPolicy(f"http://127.0.0.1:{self.port}/v1", "policy")

    def __exit__(self, *_) -> None:
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=30)
        if self.log_file:
            self.log_file.close()

    def _wait_until_ready(self, timeout: int = 300) -> None:
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"Policy server failed; see {self.log_path}")
            try:
                with urlopen(url, timeout=2):
                    return
            except OSError:
                time.sleep(2)
        raise TimeoutError(f"Policy server did not start; see {self.log_path}")
