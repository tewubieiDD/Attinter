import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


DEFAULT_GROUP_KEYS = ("dataset", "evaluation", "model", "seed", "lr", "wd")


def _read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_best_metric(info):
    summary = info.get("summary", {})
    for key in ("best_test_bacc", "best_test_acc", "best_acc"):
        value = _as_float(summary.get(key))
        if value is not None:
            return value, key

    history = info.get("history", {})
    test_history = history.get("test", {})
    for key in ("bacc", "acc"):
        values = test_history.get(key, [])
        values = [_as_float(value) for value in values]
        values = [value for value in values if value is not None]
        if values:
            return values[-1], f"history.test_{key}"

    return None, None


def _split_fields(split_info):
    fields = {
        "subject": "",
        "test_session": "",
        "test_session_name": "",
        "test_subject": "",
        "fold": "",
    }
    for key in fields:
        if key in split_info:
            fields[key] = split_info[key]
    return fields


def _iter_runs(roots, include_unfinished=False):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue

        for info_path in root.rglob("info.json"):
            run_dir = info_path.parent
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue

            try:
                config = _read_json(config_path)
                info = _read_json(info_path)
            except (OSError, json.JSONDecodeError):
                continue

            summary = info.get("summary", {})
            if not include_unfinished and summary.get("is_finished") is not True:
                continue

            best_metric, metric_source = _get_best_metric(info)
            if best_metric is None:
                continue

            split_info = config.get("split_info", {})
            row = {
                "run_dir": str(run_dir),
                "dataset": config.get("dataset", ""),
                "evaluation": config.get("evaluation", ""),
                "model": config.get("model", ""),
                "seed": config.get("seed", ""),
                "lr": config.get("lr", ""),
                "wd": config.get("wd", ""),
                "epochs": config.get("epochs", ""),
                "bs": config.get("bs", ""),
                "select_metric": config.get("select_metric", ""),
                "metric_source": metric_source,
                "best_test_bacc": best_metric,
            }
            row.update(_split_fields(split_info))
            yield row


def _group_rows(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    return groups


def _summarize(rows, group_keys):
    summary_rows = []
    for key, group in sorted(_group_rows(rows, group_keys).items()):
        values = [row["best_test_bacc"] for row in group]
        row = {group_key: value for group_key, value in zip(group_keys, key)}
        row.update(
            {
                "n": len(values),
                "mean_bacc": mean(values),
                "std_bacc": stdev(values) if len(values) > 1 else 0.0,
                "min_bacc": min(values),
                "max_bacc": max(values),
            }
        )
        summary_rows.append(row)
    return summary_rows


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _print_summary(summary_rows, group_keys):
    if not summary_rows:
        print("No finished runs found.")
        return

    for row in summary_rows:
        label = " | ".join(f"{key}={row.get(key, '')}" for key in group_keys)
        print(
            f"{label} | n={row['n']} | "
            f"mean={row['mean_bacc']:.4f} | std={row['std_bacc']:.4f}"
        )


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=["outputs"],
        help="Output roots to scan recursively.",
    )
    ap.add_argument("--detailed_csv", default="results_detailed.csv")
    ap.add_argument("--summary_csv", default="results_summary.csv")
    ap.add_argument(
        "--group_by",
        nargs="+",
        default=list(DEFAULT_GROUP_KEYS),
        help="Fields used to compute mean/std.",
    )
    ap.add_argument("--include_unfinished", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = list(_iter_runs(args.roots, include_unfinished=args.include_unfinished))
    summary_rows = _summarize(rows, args.group_by)

    detailed_fields = [
        "dataset",
        "evaluation",
        "model",
        "seed",
        "lr",
        "wd",
        "epochs",
        "bs",
        "select_metric",
        "subject",
        "test_session",
        "test_session_name",
        "test_subject",
        "fold",
        "best_test_bacc",
        "metric_source",
        "run_dir",
    ]
    summary_fields = list(args.group_by) + ["n", "mean_bacc", "std_bacc", "min_bacc", "max_bacc"]

    _write_csv(args.detailed_csv, rows, detailed_fields)
    _write_csv(args.summary_csv, summary_rows, summary_fields)
    _print_summary(summary_rows, args.group_by)

    print(f"Wrote detailed results to {args.detailed_csv}")
    print(f"Wrote summary results to {args.summary_csv}")
