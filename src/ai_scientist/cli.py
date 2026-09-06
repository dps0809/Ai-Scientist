import argparse
from pathlib import Path

from ai_scientist.pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Scientist research loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic local experiment")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    demo.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.command == "demo":
        report = run_demo(args.output_dir, args.seed)
        print(f"{report.evaluation.status}: {report.evaluation.explanation}")
        print(f"Reports written to {args.output_dir}")
