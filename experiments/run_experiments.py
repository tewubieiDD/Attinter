"""Build, schedule and run independent BNCI UDA experiments."""
import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from dataloader import make_dataloader, merge_records
from spd.models import build_model
from utils import Recorder, create_exp_dir, set_seed
from utils.train_bacc_uda import train_network_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dataset", choices=("BNCI2014001", "BNCI2015001"), default="BNCI2014001")
    p.add_argument("--evaluation", choices=("inter-session", "inter-subject"), default="inter-session")
    p.add_argument("--model", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--subject", "--sub", dest="subject", type=int, default=1)
    p.add_argument("--test_session", type=str, default="all", help="T/E, 1/2, or all")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", "--bs", dest="batch_size", type=int, default=128)
    p.add_argument("--validation_size", type=float, default=.2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-1)
    p.add_argument("--slice", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--loader_workers", type=int, default=0)
    p.add_argument("--no_channel_dim", action="store_true")
    p.add_argument("--description", type=str, default="")
    return p.parse_args()


def dataset_module(dataset):
    return __import__(f"dataloader.{dataset.lower()}_dataloader", fromlist=["*"])


def build_inter_session_experiments(args, module, data_path):
    subjects = [args.subject] if args.subject > 0 else module.available_subjects(data_path)
    requested = None if args.test_session.lower() == "all" else {module.session_id(args.test_session)}
    configs = []
    for subject in subjects:
        sessions = module.available_sessions(data_path, subject)
        for target in sessions:
            if requested is not None and target not in requested: continue
            source = [s for s in sessions if s != target]
            configs.append({"id": f"sub{subject:02d}_target{module.session_name(target)}",
                            "evaluation": "inter-session", "source_pairs": [(subject, s) for s in source],
                            "target_pairs": [(subject, target)], "domain_key": "domain_inter_session"})
    return configs


def build_inter_subject_experiments(args, module, data_path):
    subjects = [args.subject] if args.subject > 0 else module.available_subjects(data_path)
    configs = []
    for target_subject in subjects:
        source_subjects = [s for s in subjects if s != target_subject]
        configs.append({"id": f"target_sub{target_subject:02d}", "evaluation": "inter-subject",
                        "source_pairs": [(s, se) for s in source_subjects for se in
                                         module.available_sessions(data_path, s)],
                        "target_pairs": [(target_subject, se) for se in
                                         module.available_sessions(data_path, target_subject)],
                        "domain_key": "domain_inter_subject"})
    return configs


def build_experiments(args):
    module = dataset_module(args.dataset)
    data_path = args.data_path or str(module.DEFAULT_DATA_PATH)
    if args.evaluation == "inter-session":
        configs = build_inter_session_experiments(args, module, data_path)
    else:
        configs = build_inter_subject_experiments(args, module, data_path)
    for config in configs:
        config.update({"dataset": args.dataset, "data_path": data_path, "batch_size": args.batch_size,
                       "validation_size": args.validation_size, "seed": args.seed,
                       "add_channel_dim": not args.no_channel_dim, "loader_workers": args.loader_workers})
    return configs


def split_source(indices, y, d, validation_size, seed=42):
    indices = np.asarray(indices, dtype=np.int64)
    if validation_size <= 0: return indices, np.empty(0, dtype=np.int64)
    classes = max(1, len(np.unique(y)))
    strat = y[indices] + d[indices] * classes
    try:
        a, b = next(StratifiedShuffleSplit(1, test_size=validation_size, random_state=seed).split(indices, strat))
    except ValueError:
        a, b = train_test_split(np.arange(len(indices)), test_size=validation_size, random_state=seed,
                                stratify=y[indices] if len(np.unique(y[indices])) > 1 else None)
    return indices[a], indices[b]


def load_records(module, data_path, pairs):
    return [module.load_subject_session(data_path, subject, session) for subject, session in pairs]


def run_one_experiment(config, args):
    torch.set_num_threads(1)
    set_seed(args.seed)
    module = dataset_module(config["dataset"])
    source_records = load_records(module, config["data_path"], config["source_pairs"])
    target_records = load_records(module, config["data_path"], config["target_pairs"])
    source = merge_records(source_records)
    target = merge_records(target_records)
    source_indices = np.arange(len(source["y"]))
    train_idx, val_idx = split_source(source_indices, source["y"], source[config["domain_key"]],
                                      config["validation_size"])
    target_indices = np.arange(len(target["y"]))
    train_loader = make_dataloader(source, train_idx, config["domain_key"], config["batch_size"], True,
                                   config["add_channel_dim"], config["loader_workers"])
    val_loader = make_dataloader(source, val_idx, config["domain_key"], config["batch_size"], False,
                                 config["add_channel_dim"], config["loader_workers"])
    test_loader = make_dataloader(target, target_indices, config["domain_key"], config["batch_size"], False,
                                  config["add_channel_dim"], config["loader_workers"])
    split_info = {"experiment_id": config["id"], "evaluation": config["evaluation"],
                  "source_pairs": config["source_pairs"], "target_pairs": config["target_pairs"],
                  "source_train_size": len(train_idx), "source_val_size": len(val_idx),
                  "target_size": len(target_indices)}
    exp_dir = create_exp_dir(args.output_dir, args.seed, args.dataset, args.evaluation, config["id"], args.model,
                             args.lr, args.wd)
    recorder = Recorder(exp_dir, {**vars(args), "split_info": split_info})
    net = build_model(args.model, args.dataset, vars(args)).cpu()
    score, details = train_network_loss(net, train_loader, val_loader, test_loader, recorder,
                                        device=args.device, epochs=args.epochs, lr=args.lr, wd=args.wd)
    recorder.save_summary(score)
    return {"split_info": split_info, "target_bacc": score, "details": details}


def main():
    args = parse_args()
    configs = build_experiments(args)
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context("spawn")) as pool:
        future_to_config = {
            pool.submit(run_one_experiment, config, args): config
            for config in configs
        }
        for completed_count, future in enumerate(as_completed(future_to_config), start=1):
            config = future_to_config[future]
            try:
                results.append(future.result())
                print(
                    f"finished {completed_count}/{len(configs)} {config['id']}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "experiment_id": config.get("id"),
                    "evaluation": config.get("evaluation"),
                    "source_pairs": config.get("source_pairs"),
                    "target_pairs": config.get("target_pairs"),
                    "error": repr(exc),
                }
                failures.append(failure)
                print(
                    f"failed {completed_count}/{len(configs)} "
                    f"{config.get('id')}: {exc}",
                    flush=True,
                )
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "evaluation": args.evaluation,
        "total": len(configs),
        "succeeded": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    (out / f"summary_{args.dataset}_{args.evaluation}_{args.model}.json").write_text(
        json.dumps(summary, default=str, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
