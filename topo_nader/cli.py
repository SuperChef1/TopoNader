from __future__ import annotations

import argparse
import json
from pathlib import Path

from .architecture.spec import TopoNaderArchitectureSpec
from .architecture.validator import TopoNaderArchitectureValidator
from .config import load_config
from .pipeline.engine import TopoNaderEngine
from .pipeline.state import TopoNaderRunState


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toponader", description="TopoNader topology-surrogate optimization")
    parser.add_argument("--config", default="configs/default.json", help="TopoNader JSON configuration")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("validate-config")
    subcommands.add_parser("resolved-config")

    initialize = subcommands.add_parser("init")
    initialize.add_argument("--run-dir", required=True)
    initialize.add_argument("--force", action="store_true")

    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--iteration", required=True, type=int)

    feedback = subcommands.add_parser("feedback")
    feedback.add_argument("--run-dir", required=True)
    feedback.add_argument("--dataset", required=True)
    feedback.add_argument("--iteration", required=True, type=int)
    feedback.add_argument("--fe-result", required=True)
    feedback.add_argument("--next-dataset", required=True)

    status = subcommands.add_parser("status")
    status.add_argument("--run-dir", required=True)

    architecture = subcommands.add_parser("validate-architecture")
    architecture.add_argument("--architecture", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent
    engine = TopoNaderEngine(config, project_root)
    if args.command == "validate-config":
        print(json.dumps({"valid": True, "experiment_name": config.experiment_name}, ensure_ascii=False))
    elif args.command == "resolved-config":
        print(json.dumps(config.to_dict(), ensure_ascii=False))
    elif args.command == "init":
        state = engine.initialize_run(args.run_dir, force=args.force)
        print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        request = engine.prepare_iteration(args.run_dir, args.dataset, args.iteration)
        print(str(request.resolve()))
    elif args.command == "feedback":
        dataset = engine.apply_feedback(
            args.run_dir, args.dataset, args.iteration, args.fe_result, args.next_dataset
        )
        print(str(dataset.resolve()))
    elif args.command == "status":
        state = TopoNaderRunState.load(Path(args.run_dir) / "state.json")
        print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "validate-architecture":
        spec = TopoNaderArchitectureSpec.load(args.architecture)
        print(json.dumps(TopoNaderArchitectureValidator(config.architecture).validate(spec).to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
