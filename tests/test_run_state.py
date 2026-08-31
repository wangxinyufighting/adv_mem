import json
import tempfile
import unittest
from pathlib import Path

from training.run_state import RunState


class RunStateTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_state.json"
            state = RunState.new("model", [0, 1])
            state.next_round = 2
            state.save(path)
            restored = RunState.load(path, "other", [0, 1])
        self.assertEqual(restored.next_round, 2)
        self.assertEqual(restored.attacker_model, "model")
        self.assertEqual(set(restored.cases), {0, 1})

    def test_old_pipeline_requires_a_new_work_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_state.json"
            path.write_text(json.dumps({"next_round": 3}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new --work-dir"):
                RunState.load(path, "model", [0])

    def test_bank_cases_must_stay_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_state.json"
            RunState.new("model", [0]).save(path)
            with self.assertRaisesRegex(ValueError, "Probe Bank cases"):
                RunState.load(path, "model", [0, 1])


if __name__ == "__main__":
    unittest.main()
