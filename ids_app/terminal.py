from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from .classical import train_classical_suite
from .config import CLASSICAL_PROFILES, DNN_PROFILES, REPORTS_DIR, ROOT_DIR
from .data import dataset_summary
from .legacy import evaluate_legacy_predictions
from .storage import ensure_directories, list_run_summaries


BINARY_LABELS = {"0": "normal", "1": "attack"}
FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


AUTO_TRAIN_PROFILES = {
    "quick": {
        "classical": {"profile": "fast", "train_sample": 5000, "test_sample": 2000, "random_state": 42},
        "dnn": {
            "profile": "fast",
            "architectures": [1],
            "epochs": 1,
            "train_sample": 5000,
            "test_sample": 2000,
            "random_state": 42,
        },
    },
    "standard": {
        "classical": {"profile": "fast", "random_state": 42},
        "dnn": {"profile": "fast", "random_state": 42},
    },
    "full": {
        "classical": {"profile": "full", "random_state": 42},
        "dnn": {"profile": "full", "random_state": 42},
    },
}


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _percent(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    text_rows = [[_format_number(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in text_rows)) if text_rows else len(header)
        for index, header in enumerate(headers)
    ]
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    rule = "  ".join("-" * width for width in widths)
    body = ["  ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in text_rows]
    return "\n".join([line, rule, *body])


def _section(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=False))


def show_datasets(json_output: bool = False) -> None:
    payload = dataset_summary()
    if json_output:
        _print_json(payload)
        return

    _section("Datasets")
    rows = []
    for key in ("classical_train", "classical_test", "dnn_train", "dnn_test"):
        item = payload[key]
        labels = ", ".join(f"{label}:{count:,}" for label, count in item["label_counts"].items())
        rows.append([item["path"], item["rows"], item["columns"], f"{item['size_mb']} MB", labels])
    print(_table(["Path", "Rows", "Cols", "Size", "Labels"], rows))
    print()
    print(f"Train files match: {payload['duplicates']['train_files_match']}")
    print(f"Test files match:  {payload['duplicates']['test_files_match']}")


def show_attacks(json_output: bool = False) -> None:
    payload = dataset_summary()
    rows = []
    for key in ("classical_train", "classical_test"):
        item = payload[key]
        total = item["rows"]
        for label, count in sorted(item["label_counts"].items()):
            rows.append(
                [
                    item["path"],
                    label,
                    BINARY_LABELS.get(label, "unknown"),
                    count,
                    f"{(count / total) * 100:.2f}%",
                ]
            )

    if json_output:
        _print_json(
            {
                "label_meaning": BINARY_LABELS,
                "note": "This dataset is binary encoded. Named attack families are not present in the CSV files.",
                "rows": rows,
            }
        )
        return

    _section("Attack Label Distribution")
    print(_table(["Dataset", "Label", "Meaning", "Rows", "Share"], rows))
    print()
    print("The CSV files only contain binary labels here. The app treats 0 as normal and 1 as attack.")


def show_features(json_output: bool = False) -> None:
    rows = [[0, "label", "binary target: 0=normal, 1=attack"]]
    rows.extend([[index, name, "numeric/coded IDS feature"] for index, name in enumerate(FEATURE_NAMES, start=1)])
    if json_output:
        _print_json({"columns": rows})
        return
    _section("Feature Columns")
    print(_table(["Column", "Name", "Meaning"], rows))


def show_legacy(json_output: bool = False) -> None:
    payload = evaluate_legacy_predictions()
    if json_output:
        _print_json(payload)
        return

    _section("Legacy Classical Results")
    classical_rows = [
        [
            item["label"],
            _percent(item["metrics"]["accuracy"]),
            _percent(item["metrics"]["precision"]),
            _percent(item["metrics"]["recall"]),
            _percent(item["metrics"]["f1"]),
        ]
        for item in payload["classical"]
    ]
    print(_table(["Model", "Accuracy", "Precision", "Recall", "F1"], classical_rows))

    _section("Legacy DNN Results")
    dnn_rows = [
        [
            item["label"],
            _percent(item["metrics"]["accuracy"]),
            _percent(item["metrics"]["precision"]),
            _percent(item["metrics"]["recall"]),
            _percent(item["metrics"]["f1"]),
            item["history"]["epochs_logged"] if item.get("history") else "n/a",
        ]
        for item in payload["dnn"]
    ]
    print(_table(["Model", "Accuracy", "Precision", "Recall", "F1", "Epochs"], dnn_rows))


def show_runs(json_output: bool = False, limit: int = 20) -> None:
    payload = list_run_summaries(limit=limit)
    if json_output:
        _print_json(payload)
        return

    _section("Completed Runs")
    if not payload:
        print("No completed runs yet.")
        return

    rows = []
    for run in payload:
        best = run.get("results", [{}])[0]
        metrics = best.get("metrics", {})
        rows.append(
            [
                run.get("run_id"),
                run.get("kind"),
                best.get("label", "n/a"),
                _percent(metrics.get("accuracy", 0)),
                _percent(metrics.get("f1", 0)),
                run.get("dataset", {}).get("train_rows", 0),
                run.get("dataset", {}).get("test_rows", 0),
            ]
        )
    print(_table(["Run", "Kind", "Best Model", "Accuracy", "F1", "Train", "Test"], rows))


def show_reports(json_output: bool = False) -> None:
    reports = []
    if REPORTS_DIR.exists():
        for path in sorted(REPORTS_DIR.glob("*")):
            if path.is_file():
                reports.append(
                    {
                        "name": path.name,
                        "path": str(path.relative_to(ROOT_DIR)),
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    }
                )

    if json_output:
        _print_json(reports)
        return

    _section("Reports")
    if not reports:
        print("No report files found.")
        return
    print(_table(["Name", "Path", "Size"], [[item["name"], item["path"], f"{item['size_mb']} MB"] for item in reports]))


def show_best(json_output: bool = False) -> None:
    legacy = evaluate_legacy_predictions()
    runs = list_run_summaries(limit=50)
    best_legacy_classical = legacy["classical"][0] if legacy["classical"] else None
    best_legacy_dnn = legacy["dnn"][0] if legacy["dnn"] else None
    best_run = None
    for run in runs:
        if not run.get("results"):
            continue
        candidate = {"run_id": run["run_id"], "kind": run["kind"], **run["results"][0]}
        if best_run is None or candidate["metrics"]["f1"] > best_run["metrics"]["f1"]:
            best_run = candidate

    payload = {
        "legacy_classical": best_legacy_classical,
        "legacy_dnn": best_legacy_dnn,
        "automation_run": best_run,
    }
    if json_output:
        _print_json(payload)
        return

    _section("Best Known Results")
    rows = []
    if best_legacy_classical:
        rows.append(["Legacy Classical", best_legacy_classical["label"], _percent(best_legacy_classical["metrics"]["accuracy"]), _percent(best_legacy_classical["metrics"]["f1"])])
    if best_legacy_dnn:
        rows.append(["Legacy DNN", best_legacy_dnn["label"], _percent(best_legacy_dnn["metrics"]["accuracy"]), _percent(best_legacy_dnn["metrics"]["f1"])])
    if best_run:
        rows.append([f"Run {best_run['run_id']}", best_run["label"], _percent(best_run["metrics"]["accuracy"]), _percent(best_run["metrics"]["f1"])])
    print(_table(["Source", "Model", "Accuracy", "F1"], rows))


def show_head(path_text: str, lines: int = 5) -> None:
    path = _resolve_repo_path(path_text)
    if not path.exists() or not path.is_file():
        print(f"Not a file: {path_text}")
        return
    if path.stat().st_size > 100 * 1024 * 1024:
        print("Refusing to print a very large file. Use a smaller file or a sampled CSV command.")
        return
    _section(f"Head: {path.relative_to(ROOT_DIR)}")
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for index, line in enumerate(handle):
            if index >= lines:
                break
            print(line.rstrip())


def show_overview(json_output: bool = False) -> None:
    payload = {
        "datasets": dataset_summary(),
        "legacy": evaluate_legacy_predictions(),
        "runs": list_run_summaries(limit=8),
        "profiles": {"classical": CLASSICAL_PROFILES, "dnn": DNN_PROFILES},
    }
    if json_output:
        _print_json(payload)
        return
    show_datasets()
    show_attacks()
    show_legacy()
    show_runs(limit=8)
    show_reports()


def train_classical(args: argparse.Namespace) -> None:
    config: dict[str, Any] = {"profile": args.profile, "random_state": args.random_state}
    if args.train_sample is not None:
        config["train_sample"] = args.train_sample
    if args.test_sample is not None:
        config["test_sample"] = args.test_sample
    if args.models:
        config["models"] = args.models

    _section("Running Classical Train/Test")
    summary = train_classical_suite("terminal", config)
    _print_train_summary(summary)


def train_dnn(args: argparse.Namespace) -> None:
    from .dnn import train_dnn_suite

    config: dict[str, Any] = {"profile": args.profile, "random_state": args.random_state}
    if args.architectures:
        config["architectures"] = args.architectures
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.train_sample is not None:
        config["train_sample"] = args.train_sample
    if args.test_sample is not None:
        config["test_sample"] = args.test_sample

    _section("Running DNN Train/Test")
    summary = train_dnn_suite("terminal", config)
    _print_train_summary(summary)


def auto_train(args: argparse.Namespace) -> None:
    profile_name = args.auto_profile
    config = AUTO_TRAIN_PROFILES[profile_name]
    _section(f"Auto Train: {profile_name}")
    print("Running classical suite...")
    classical_summary = train_classical_suite("terminal-auto", dict(config["classical"]))
    _print_train_summary(classical_summary)

    if args.skip_dnn:
        return

    print()
    print("Running DNN suite...")
    from .dnn import train_dnn_suite

    dnn_summary = train_dnn_suite("terminal-auto", dict(config["dnn"]))
    _print_train_summary(dnn_summary)


def _print_train_summary(summary: dict[str, Any]) -> None:
    print(f"Run ID: {summary['run_id']}")
    print(f"Train rows: {summary['dataset']['train_rows']:,}")
    print(f"Test rows:  {summary['dataset']['test_rows']:,}")
    rows = []
    for item in summary["results"]:
        rows.append(
            [
                item["label"],
                f"{item['training_seconds']}s",
                _percent(item["metrics"]["accuracy"]),
                _percent(item["metrics"]["precision"]),
                _percent(item["metrics"]["recall"]),
                _percent(item["metrics"]["f1"]),
            ]
        )
    print()
    print(_table(["Model", "Time", "Accuracy", "Precision", "Recall", "F1"], rows))
    print()
    print(f"Saved summary: automation/runs/{summary['run_id']}/summary.json")


def command_shell() -> None:
    ensure_directories()
    print("IDS command console. Type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            raw = input("ids> ").strip()
        except EOFError:
            print()
            return
        if not raw:
            continue
        try:
            should_exit = run_shell_command(raw)
        except Exception as exc:
            print(f"error: {exc}")
            continue
        if should_exit:
            return


def run_shell_command(raw: str) -> bool:
    parts = shlex.split(raw)
    if not parts:
        return False
    command, *args = parts
    command = command.lower()

    if command in {"exit", "quit", "q"}:
        return True
    if command == "help":
        print_shell_help()
    elif command == "clear":
        os.system("clear")
    elif command == "pwd":
        print(ROOT_DIR)
    elif command == "ls":
        list_path(args[0] if args else ".")
    elif command == "head":
        if not args:
            print("usage: head <path> [lines]")
        else:
            show_head(args[0], int(args[1]) if len(args) > 1 else 5)
    elif command in {"overview", "status"}:
        show_overview()
    elif command in {"data", "datasets"}:
        show_datasets()
    elif command in {"attack", "attacks"}:
        show_attacks()
    elif command in {"feature", "features", "schema"}:
        show_features()
    elif command == "legacy":
        show_legacy()
    elif command == "reports":
        show_reports()
    elif command == "runs":
        show_runs(limit=int(args[0]) if args else 20)
    elif command == "best":
        show_best()
    elif command == "train":
        run_train_command(args)
    else:
        print(f"Unknown command: {command}. Type 'help'.")
    return False


def print_shell_help() -> None:
    _section("Commands")
    rows = [
        ["help", "show this command list"],
        ["overview | status", "show datasets, attacks, legacy metrics, runs, reports"],
        ["datasets | data", "show CSV sizes, labels, duplicate checks"],
        ["attacks", "show attack/normal label distribution"],
        ["features | schema", "show the 42 dataset columns"],
        ["legacy", "show saved legacy prediction metrics"],
        ["runs [limit]", "show completed train/test runs"],
        ["best", "show best legacy and automation results"],
        ["reports", "list report PDFs"],
        ["ls [path]", "list repo files"],
        ["head <path> [lines]", "print first lines of a text/CSV file"],
        ["train auto [quick|standard|full]", "train classical and DNN automatically"],
        ["train classical [quick|fast|balanced|full]", "run classical train/test"],
        ["train dnn [quick|fast|balanced|full]", "run DNN train/test"],
        ["clear", "clear terminal"],
        ["exit", "quit"],
    ]
    print(_table(["Command", "Action"], rows))


def run_train_command(args: list[str]) -> None:
    if not args:
        print("usage: train <auto|classical|dnn> [quick|fast|standard|balanced|full]")
        return
    kind = args[0].lower()
    profile = args[1].lower() if len(args) > 1 else "quick"

    if kind == "auto":
        if profile == "fast":
            profile = "quick"
        if profile not in AUTO_TRAIN_PROFILES:
            print("profiles: quick, standard, full")
            return
        namespace = argparse.Namespace(auto_profile=profile, skip_dnn=False)
        auto_train(namespace)
    elif kind == "classical":
        if profile == "quick":
            namespace = argparse.Namespace(profile="fast", train_sample=5000, test_sample=2000, models=None, random_state=42)
        else:
            if profile == "standard":
                profile = "fast"
            if profile not in CLASSICAL_PROFILES:
                print("profiles: quick, fast, balanced, full")
                return
            namespace = argparse.Namespace(profile=profile, train_sample=None, test_sample=None, models=None, random_state=42)
        train_classical(namespace)
    elif kind == "dnn":
        if profile == "quick":
            namespace = argparse.Namespace(profile="fast", architectures=[1], epochs=1, batch_size=128, train_sample=5000, test_sample=2000, random_state=42)
        else:
            if profile == "standard":
                profile = "fast"
            if profile not in DNN_PROFILES:
                print("profiles: quick, fast, balanced, full")
                return
            namespace = argparse.Namespace(profile=profile, architectures=None, epochs=None, batch_size=None, train_sample=None, test_sample=None, random_state=42)
        train_dnn(namespace)
    else:
        print("train kinds: auto, classical, dnn")


def list_path(path_text: str) -> None:
    path = _resolve_repo_path(path_text)
    if not path.exists():
        print(f"Not found: {path_text}")
        return
    if path.is_file():
        print(path.relative_to(ROOT_DIR))
        return
    entries = []
    for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
        marker = "/" if child.is_dir() else ""
        entries.append([child.name + marker, f"{child.stat().st_size:,}" if child.is_file() else "dir"])
    print(_table(["Name", "Size"], entries))


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT_DIR)
    except ValueError:
        raise ValueError("path must stay inside the project directory")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal-only IDS automation console.")
    parser.add_argument("--json", action="store_true", help="Print JSON for commands that support it.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("overview", help="Show datasets, legacy metrics, runs, and reports.")
    subparsers.add_parser("datasets", help="Show CSV file sizes, labels, hashes, and duplicate checks.")
    subparsers.add_parser("attacks", help="Show attack/normal label distribution.")
    subparsers.add_parser("features", help="Show target and feature columns.")
    subparsers.add_parser("legacy", help="Show metrics for existing saved prediction files.")
    subparsers.add_parser("best", help="Show best known legacy and generated results.")
    subparsers.add_parser("shell", help="Open the interactive IDS command shell.")
    runs_parser = subparsers.add_parser("runs", help="Show completed automation runs.")
    runs_parser.add_argument("--limit", type=int, default=20)
    subparsers.add_parser("reports", help="List PDF reports available in the repo.")

    auto_parser = subparsers.add_parser("auto-train", help="Train classical and DNN suites automatically.")
    auto_parser.add_argument("--auto-profile", choices=AUTO_TRAIN_PROFILES.keys(), default="quick")
    auto_parser.add_argument("--skip-dnn", action="store_true")

    classical_parser = subparsers.add_parser("train-classical", help="Run a classical ML train/test suite.")
    classical_parser.add_argument("--profile", choices=CLASSICAL_PROFILES.keys(), default="fast")
    classical_parser.add_argument("--train-sample", type=int)
    classical_parser.add_argument("--test-sample", type=int)
    classical_parser.add_argument("--models", nargs="+")
    classical_parser.add_argument("--random-state", type=int, default=42)

    dnn_parser = subparsers.add_parser("train-dnn", help="Run a DNN train/test suite.")
    dnn_parser.add_argument("--profile", choices=DNN_PROFILES.keys(), default="fast")
    dnn_parser.add_argument("--architectures", nargs="+", type=int)
    dnn_parser.add_argument("--epochs", type=int)
    dnn_parser.add_argument("--batch-size", type=int)
    dnn_parser.add_argument("--train-sample", type=int)
    dnn_parser.add_argument("--test-sample", type=int)
    dnn_parser.add_argument("--random-state", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_directories()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command is None:
            command_shell()
        elif args.command == "shell":
            command_shell()
        elif args.command == "overview":
            show_overview(args.json)
        elif args.command == "datasets":
            show_datasets(args.json)
        elif args.command == "attacks":
            show_attacks(args.json)
        elif args.command == "features":
            show_features(args.json)
        elif args.command == "legacy":
            show_legacy(args.json)
        elif args.command == "best":
            show_best(args.json)
        elif args.command == "runs":
            show_runs(args.json, args.limit)
        elif args.command == "reports":
            show_reports(args.json)
        elif args.command == "auto-train":
            auto_train(args)
        elif args.command == "train-classical":
            train_classical(args)
        elif args.command == "train-dnn":
            train_dnn(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
