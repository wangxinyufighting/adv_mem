import argparse
from pathlib import Path

from attacker.answer_agent import QwenAnswerAgent
from attacker.oracle import DeepSeekOracle
from attacker.probe import FixedProbeQuestionGenerator, ProbeFactory
from attacker.probe_bank import ProbeBank
from attacker.reward_judge import DeepSeekRewardJudge
from training.dataset_builder import RouteProposalBuilder
from utils.longmemeval_graph_reader import LongMemEvalGraphReader


ROOT = Path(__file__).resolve().parents[1]


def build(args: argparse.Namespace) -> ProbeBank:
    reader = LongMemEvalGraphReader(args.graph, args.longmemeval, args.graph_version)
    cases = {}
    if args.output.exists():
        old = ProbeBank.load(args.output)
        if old.graph_version != args.graph_version:
            raise ValueError("Probe Bank graph version does not match")
        cases.update(old.cases)

    factory = ProbeFactory(
        FixedProbeQuestionGenerator.from_env(),
        DeepSeekOracle(),
        QwenAnswerAgent.from_env(),
        DeepSeekRewardJudge.from_env(),
    )
    for case_index in reader.available_cases():
        probes = list(cases.get(case_index, ()))
        if len(probes) >= args.probes_per_case:
            continue
        signatures = {probe.route.route_signature for probe in probes}
        proposed = 0
        batch = 0
        while (
            len(probes) < args.probes_per_case
            and proposed < args.max_routes_per_case
        ):
            count = min(
                args.routes_per_batch,
                args.max_routes_per_case - proposed,
            )
            routes = RouteProposalBuilder(
                reader,
                args.seed + case_index * args.max_routes_per_case + batch,
            ).routes(case_index, count)
            routes = tuple(
                route for route in routes if route.route_signature not in signatures
            )
            proposed += count
            batch += 1
            signatures.update(route.route_signature for route in routes)
            probes.extend(factory.build_many(routes))
            print(
                f"Probe Bank, case {case_index}: "
                f"proposed={proposed} valid={len(probes)}",
                flush=True,
            )
        if len(probes) < 2:
            raise RuntimeError(f"Case {case_index} produced fewer than two probes")
        cases[case_index] = tuple(probes[: args.probes_per_case])
        ProbeBank(args.graph_version, cases).save(args.output)
    return ProbeBank(args.graph_version, cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen Probe Bank")
    parser.add_argument(
        "--graph",
        default=ROOT / "data/longmemeval/memory_graph_fullgraph5.json",
    )
    parser.add_argument(
        "--longmemeval",
        default=ROOT / "data/longmemeval/longmemeval_s.json",
    )
    parser.add_argument("--graph-version", default="fullgraph5")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/longmemeval/probe_bank_fullgraph5.json",
    )
    parser.add_argument("--probes-per-case", type=int, default=32)
    parser.add_argument("--routes-per-batch", type=int, default=16)
    parser.add_argument("--max-routes-per-case", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = build(args)
    print(
        f"Saved {sum(len(items) for items in bank.cases.values())} probes "
        f"across {len(bank.cases)} cases to {args.output}"
    )


if __name__ == "__main__":
    main()
