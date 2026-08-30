import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.APIError = type("APIError", (Exception,), {})
    sys.modules["openai"] = openai
if "pyarrow" not in sys.modules:
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.Table = object
    sys.modules["pyarrow"] = pyarrow
    sys.modules["pyarrow.parquet"] = types.ModuleType("pyarrow.parquet")

from memory.models import MemoryState
from training.run_state import RunState


class RunStateMigrationTests(unittest.TestCase):
    def test_old_question_attacker_checkpoint_is_not_reused(self):
        memory = MemoryState.empty().to_dict()
        memory.pop("attack_history")
        payload = {
            "next_round": 3,
            "attacker_model": "old-question-attacker",
            "builder_model": "old-builder",
            "cases": {
                "0": {
                    "memory": memory,
                    "questions": [],
                    "stop_state": {"converged_rounds": 0},
                    "stopped": False,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_state.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            state = RunState.load(path, "base-route-selector", [0])
            self.assertEqual(state.attacker_model, "base-route-selector")
            self.assertEqual(state.builder_model, "old-builder")
            self.assertEqual(state.attacker_role, "route_selector_v1")
            self.assertEqual(state.cases[0].memory.attack_history, {})

            state.save(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["attacker_role"], "route_selector_v1")


if __name__ == "__main__":
    unittest.main()
