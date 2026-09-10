"""Random search for joint BERT/KGE K-vs-all training hyperparameters.

Example (12 trials sampled from 288 default combinations)::

    python -m dicee.scripts.bert_bce_kvsall_link_prediction_random_search \
        --dataset_path bert_datasets/nell-995-sem/0.0 \
        --bert_model_path checkpoints/nell-995-h100/bert-pretrained \
        --kge_model TransE --kge_num_epochs_grid 100 200 \
        --bert_num_epochs_grid 10 20 30 --random_trials 12

Use --random_trials 288 for the complete default space, or --dry_run to preview.
All other training options come from bert_bce_kvsall_link_prediction.py.
Trials use fixed epoch budgets and final validation, and the
search selects the highest validation MRR. Test ranking is never run here.
KGE and BERT epoch budgets are sampled independently.
New searches create random_config.json, random_results.json, best_trial.json,
and each trial's original artifacts. Continue an interrupted search with::

    python -m dicee.scripts.bert_bce_kvsall_link_prediction_random_search \
        --resume_search path/to/search_directory

Resume restores the saved arguments and trial order, skips completed trials,
and continues interrupted trials from their last saved epoch. A trial that
has not saved a checkpoint starts again. Only execution options such as
--device and --candidate_batch_size may change when resuming.
"""

import argparse
import gc
import glob
import itertools
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime

import torch

from dicee.scripts import bert_bce_kvsall_link_prediction as training


SEARCH_PARAMETERS = (
    "kge_lr",
    "kge_embedding_dim",
    "kge_num_epochs",
    "bert_num_epochs",
    "lr",
    "lambda_val",
    "negative_ratio",
)
SEARCH_SPEC = (
    ("kge_lr", float, [0.1, 0.01], "KGE learning rates"),
    ("kge_embedding_dim", int, [32, 128], "KGE embedding dimensions"),
    ("kge_num_epochs", int, [100, 200], "KGE epoch budgets"),
    ("bert_num_epochs", int, [10, 20, 30], "BERT epoch budgets"),
    ("lr", float, [2e-5, 5e-5], "BERT learning rates"),
    ("lambda_val", float, [0.3, 0.5, 0.7], "fixed BERT fusion weights"),
    ("negative_ratio", int, [5, 10], "BERT negatives per positive"),
)
SEARCH_ARGUMENTS = {
    *(f"{name}_grid" for name in SEARCH_PARAMETERS),
    "random_trials",
    "search_seed",
    "dry_run",
    "resume_search",
}
RESUME_EXECUTION_ARGUMENTS = {
    "device", "candidate_batch_size", "num_workers",
    "dry_run", "resume_search",
}


def remove_options(parser, destinations):
    """Hide trainer arguments that the search fixes or replaces with grids."""
    destinations = set(destinations)
    removed_defaults = {}
    for action in list(parser._actions):
        if action.dest not in destinations:
            continue
        removed_defaults[action.dest] = action.default
        parser._actions.remove(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)
        for option in action.option_strings:
            parser._option_string_actions.pop(option, None)
    parser.set_defaults(**removed_defaults)


def parse_args(
    argv=None, *, training_module=training, encoder="bert",
    default_output_dir="bert_bce_kvsall_random_runs",
    search_spec=SEARCH_SPEC,
    resume_execution_arguments=RESUME_EXECUTION_ARGUMENTS,
):
    argv = sys.argv[1:] if argv is None else argv
    parser = training_module.build_parser()
    search_names = tuple(item[0] for item in search_spec)
    remove_options(
        parser,
        (*search_names, "eval_split", "resume_from_checkpoint"),
    )
    parser.set_defaults(eval_split="valid", resume_from_checkpoint=None)
    parser.allow_abbrev = False
    parser.description = (
        f"Random search for joint {encoder} + KGE K-vs-all link prediction, "
        "selected by validation MRR. Only the *_grid values are searched; "
        "they override the corresponding training arguments. "
        "--output_dir is the "
        f"parent of a new search directory (default: {default_output_dir}). "
        "Use --resume_search to continue a saved search."
    )
    for name, value_type, default, description in search_spec:
        parser.add_argument(
            f"--{name}_grid", type=value_type, nargs="+", default=default,
            help=f"Candidate {description} (default: {' '.join(map(str, default))}).",
        )
    parser.add_argument(
        "--random_trials", type=int, default=12,
        help="Number of unique combinations to sample (default: 12).",
    )
    parser.add_argument(
        "--search_seed", type=int, default=None,
        help="Sampling seed; defaults to --seed. Training uses --seed in every trial.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the sampled trials without training or creating files.",
    )
    parser.add_argument(
        "--resume_search", type=str, default=None,
        help="Existing search directory; restores its configuration and progress.",
    )
    args = parser.parse_args(argv)
    if args.resume_search is not None:
        _, config = load_search_config(args.resume_search, encoder=encoder)
        supplied = {
            token.split("=", 1)[0][2:] for token in argv if token.startswith("--")
        }
        for name, value in config["args"].items():
            if name in resume_execution_arguments:
                if name not in supplied:
                    setattr(args, name, value)
            elif name in supplied and getattr(args, name) != value:
                raise ValueError(f"Cannot change --{name} when resuming a search.")
            else:
                setattr(args, name, value)
    return args


def load_search_config(path, *, encoder="bert"):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = training.resolve_path(repo_root, path)
    with open(os.path.join(output_dir, "random_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    saved_encoder = config.get("encoder", "bert")
    if saved_encoder != encoder:
        raise ValueError(
            f"Cannot resume a {saved_encoder} search with the {encoder} runner."
        )
    if saved_encoder == "bert" and config.get("training_recipe") != training.TRAINING_RECIPE:
        raise ValueError("Saved search uses a different training recipe; start a new search.")
    return output_dir, config


def validate_args(args, *, training_module=training, parameter_names=SEARCH_PARAMETERS):
    training_module.validate_args(args)
    if args.eval_split != "valid":
        raise ValueError("Random search requires --eval_split valid for selection.")
    if args.resume_from_checkpoint is not None:
        raise ValueError(
            "Use --resume_search for random search; omit --resume_from_checkpoint."
        )
    for parameter in parameter_names:
        name = f"{parameter}_grid"
        values = getattr(args, name)
        if not values or any(
            not math.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError(f"--{name} must contain finite positive values.")
        if len(set(values)) != len(values):
            raise ValueError(f"--{name} must not contain duplicate values.")
    if "lambda_val" in parameter_names and any(value >= 1.0 for value in args.lambda_val_grid):
        raise ValueError("--lambda_val_grid must be strictly between 0 and 1.")
    if args.batch_size < 1 + max(args.negative_ratio_grid):
        raise ValueError(
            "--batch_size must be at least 1 + the largest "
            "--negative_ratio_grid value for grouped encoder negative batches."
        )
    candidate_count = math.prod(
        len(getattr(args, f"{name}_grid")) for name in parameter_names
    )
    if not 1 <= args.random_trials <= candidate_count:
        raise ValueError(
            f"--random_trials must be between 1 and the number of unique "
            f"candidate combinations ({candidate_count})."
        )


def build_search(args, parameter_names=SEARCH_PARAMETERS):
    """Sample the discrete search space without changing the training RNG."""
    candidates = list(itertools.product(
        *(getattr(args, f"{name}_grid") for name in parameter_names)
    ))
    seed = args.seed if args.search_seed is None else args.search_seed
    return [
        {
            "trial_id": trial_id,
            "hyperparameters": dict(zip(parameter_names, values)),
        }
        for trial_id, values in enumerate(
            random.Random(seed).sample(candidates, args.random_trials), start=1
        )
    ]


def check_saved_args(saved, current, ignored=RESUME_EXECUTION_ARGUMENTS):
    mismatches = [
        name for name, value in saved.items()
        if name not in ignored and getattr(current, name, None) != value
    ]
    if mismatches:
        raise ValueError(f"Saved configuration mismatch: {', '.join(mismatches)}")


def validate_saved_trials(args, trials, parameter_names=SEARCH_PARAMETERS):
    if len(trials) != args.random_trials:
        raise ValueError("Saved trial count does not match random_trials.")
    seen = set()
    for trial_id, trial in enumerate(trials, start=1):
        parameters = trial["hyperparameters"]
        if trial["trial_id"] != trial_id or set(parameters) != set(parameter_names):
            raise ValueError("Invalid saved trial definition.")
        values = tuple(parameters[name] for name in parameter_names)
        if values in seen or any(
            parameters[name] not in getattr(args, f"{name}_grid")
            for name in parameter_names
        ):
            raise ValueError("Saved trials contain duplicate or invalid configurations.")
        seen.add(values)


@contextmanager
def search_lock(output_dir):
    """Prevent two processes from updating the same search simultaneously."""
    import fcntl

    with open(os.path.join(output_dir, ".search.lock"), "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This search is already running in another process.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def trial_record(trial, result, trial_args, *, encoder="bert",
                 resume_execution_arguments=RESUME_EXECUTION_ARGUMENTS):
    if encoder == "bert" and result.get("training_recipe") != training.TRAINING_RECIPE:
        raise ValueError(
            "Trial artifacts use a different training recipe; start a new search."
        )
    check_saved_args(
        result["args"], trial_args,
        resume_execution_arguments | {"output_dir", "resume_from_checkpoint"},
    )
    metrics = result["best_validation_metrics"]
    if not math.isfinite(float(metrics["MRR"])):
        raise ValueError(f"Trial {trial['trial_id']} returned a non-finite validation MRR.")
    checkpoint_path = os.path.join(
        result["output_dir"],
        f"joint_{encoder}_{trial_args.kge_model}_kvsall_link_prediction.pt",
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Completed trial checkpoint is missing: {checkpoint_path}")
    return {
        **trial,
        "training_recipe": result.get("training_recipe"),
        "validation_metrics": metrics,
        "args": result["args"],
        "output_dir": result["output_dir"],
        "checkpoint_path": checkpoint_path,
        "runtime_min": result.get("runtime_min"),
    }


def recover_trial(trial, trial_args, *, encoder="bert",
                  resume_execution_arguments=RESUME_EXECUTION_ARGUMENTS):
    """Recover completion even if interruption preceded the search summary write."""
    parent = trial_args.output_dir
    completed_path = os.path.join(parent, "completed.json")
    if os.path.isfile(completed_path):
        with open(completed_path, encoding="utf-8") as handle:
            record = json.load(handle)
        if any(record[key] != trial[key] for key in ("trial_id", "hyperparameters")):
            raise ValueError(f"Completed trial does not match trial {trial['trial_id']}.")
        trial_record(trial, {
            **record, "best_validation_metrics": record["validation_metrics"],
        }, trial_args, encoder=encoder,
           resume_execution_arguments=resume_execution_arguments)
        return record

    # Older searches also have the per-run artifacts, without completed.json.
    results = glob.glob(os.path.join(parent, "*", "results.json"))
    if results:
        result_path = max(results, key=os.path.getmtime)
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)
        result["output_dir"] = os.path.dirname(result_path)
        return trial_record(
            trial, result, trial_args, encoder=encoder,
            resume_execution_arguments=resume_execution_arguments,
        )

    # The final model is saved only after validation. Recover its metrics if
    # the process stopped before results.json (including after early stopping).
    checkpoints = glob.glob(os.path.join(
        parent, "*", f"joint_{encoder}_{trial_args.kge_model}_kvsall_link_prediction.pt"
    ))
    if checkpoints:
        checkpoint_path = max(checkpoints, key=os.path.getmtime)
        result = training.load_torch_checkpoint(checkpoint_path)
        result["output_dir"] = os.path.dirname(checkpoint_path)
        return trial_record(
            trial, result, trial_args, encoder=encoder,
            resume_execution_arguments=resume_execution_arguments,
        )

    states = glob.glob(os.path.join(parent, "*", "training_state.pt"))
    if states:
        trial_args.resume_from_checkpoint = max(states, key=os.path.getmtime)
    return None


def main(
    args=None, *, training_module=training, encoder="bert",
    default_output_dir="bert_bce_kvsall_random_runs",
    parameter_names=SEARCH_PARAMETERS,
    search_arguments=SEARCH_ARGUMENTS,
    resume_execution_arguments=RESUME_EXECUTION_ARGUMENTS,
):
    if args is None:
        args = parse_args(
            training_module=training_module, encoder=encoder,
            default_output_dir=default_output_dir,
        )
    validate_args(args, training_module=training_module, parameter_names=parameter_names)
    if args.resume_search is not None:
        output_dir, config = load_search_config(args.resume_search, encoder=encoder)
        check_saved_args(config["args"], args, resume_execution_arguments)
        trials = config["trials"]
        validate_saved_trials(args, trials, parameter_names)
    else:
        trials = build_search(args, parameter_names)
    if args.dry_run:
        print(json.dumps(trials, indent=2))
        return trials

    if args.resume_search is not None:
        with search_lock(output_dir):
            return run_trials(
                args, trials, output_dir, training_module=training_module,
                encoder=encoder, search_arguments=search_arguments,
                resume_execution_arguments=resume_execution_arguments,
            )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_parent = (
        training.resolve_path(repo_root, args.output_dir)
        if args.output_dir is not None
        else os.path.join(repo_root, default_output_dir)
    )
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{training.dataset_run_name(args.dataset_path)}_joint_kvsall_"
        f"{args.kge_model}_random_valid_{timestamp}"
    )
    if encoder != "bert":
        run_name = f"{encoder}_{run_name}"
    output_dir = training.create_unique_output_dir(output_parent, run_name)
    training.atomic_json_save(
        os.path.join(output_dir, "random_config.json"),
        {
            "encoder": encoder,
            "training_recipe": getattr(training_module, "TRAINING_RECIPE", None),
            "args": vars(args),
            "search_seed": args.seed if args.search_seed is None else args.search_seed,
            "selection_split": "valid",
            "selection_metric": "MRR",
            "trials": trials,
        },
    )
    with search_lock(output_dir):
        return run_trials(
            args, trials, output_dir, training_module=training_module,
            encoder=encoder, search_arguments=search_arguments,
            resume_execution_arguments=resume_execution_arguments,
        )


def run_trials(args, trials, output_dir, *, training_module=training, encoder="bert",
               search_arguments=SEARCH_ARGUMENTS,
               resume_execution_arguments=RESUME_EXECUTION_ARGUMENTS):
    print(f"random_search_output_dir={output_dir}")
    print(f"total_trials={len(trials)} selection_split=valid selection_metric=MRR")
    summary = {
        "training_recipe": getattr(training_module, "TRAINING_RECIPE", None),
        "selection_split": "valid",
        "selection_metric": "MRR",
        "total_trials": len(trials),
        "completed_trials": 0,
        "trials": [],
        "best_trial": None,
        "output_dir": output_dir,
    }
    for trial in trials:
        trial_args = argparse.Namespace(**{
            name: value for name, value in vars(args).items()
            if name not in search_arguments
        })
        for name, value in trial["hyperparameters"].items():
            setattr(trial_args, name, value)
        trial_args.output_dir = os.path.join(
            output_dir, "trials", f"trial_{trial['trial_id']:04d}"
        )
        record = recover_trial(
            trial, trial_args, encoder=encoder,
            resume_execution_arguments=resume_execution_arguments,
        )
        if record is None:
            action = "Resuming" if trial_args.resume_from_checkpoint else "Starting"
            print(f"{action} trial {trial['trial_id']}/{len(trials)}: {trial['hyperparameters']}")
            try:
                result = training_module.main(trial_args)
                record = trial_record(
                    trial, result, trial_args, encoder=encoder,
                    resume_execution_arguments=resume_execution_arguments,
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            print(f"Skipping completed trial {trial['trial_id']}/{len(trials)}")
        training.atomic_json_save(
            os.path.join(trial_args.output_dir, "completed.json"), record
        )
        summary["trials"].append(record)
        summary["completed_trials"] = len(summary["trials"])
        best = summary["best_trial"]
        if best is None or record["validation_metrics"]["MRR"] > best["validation_metrics"]["MRR"]:
            summary["best_trial"] = record
        # Persist after each successful trial so partial searches remain reviewable.
        training.atomic_json_save(
            os.path.join(output_dir, "random_results.json"), summary
        )
        training.atomic_json_save(
            os.path.join(output_dir, "best_trial.json"), summary["best_trial"]
        )

    best = summary["best_trial"]
    print(
        f"best_trial={best['trial_id']} valid_mrr={best['validation_metrics']['MRR']:.6f} "
        f"hyperparameters={json.dumps(best['hyperparameters'], sort_keys=True)}"
    )
    print(f"best_checkpoint={best['checkpoint_path']}")
    return summary


if __name__ == "__main__":
    main()
