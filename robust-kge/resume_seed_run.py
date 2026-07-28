"""Resume helpers for the long-running seed experiment drivers."""

import json
from pathlib import Path
from types import SimpleNamespace


def _load_trained_model(run_dir):
    """Load a model while tolerating saved state owned only by its loss."""
    import torch

    from dicee.static_funcs import intialize_model, load_json, load_model

    try:
        model, _ = load_model(str(run_dir))
        return model
    except RuntimeError as error:
        # Some data-aware losses register dataset-derived buffers during
        # training.  They are serialized below ``loss.*`` but are absent when
        # a model is initialized for inference.  They do not affect scoring.
        if "Unexpected key(s) in state_dict" not in str(error):
            raise

    config = load_json(str(run_dir / "configuration.json"))
    report = load_json(str(run_dir / "report.json"))
    config["num_entities"] = report["num_entities"]
    config["num_relations"] = report["num_relations"]
    model, _ = intialize_model(config)
    weights = torch.load(run_dir / "model.pt", map_location="cpu")
    incompatible = model.load_state_dict(weights, strict=False)

    unexpected_model_keys = [
        key for key in incompatible.unexpected_keys if not key.startswith("loss.")
    ]
    if incompatible.missing_keys or unexpected_model_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch while resuming evaluation: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected_model_keys}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def resume_seed_run(run_dir, logger):
    """Return an existing result, or finish evaluation for a trained run.

    ``None`` means that training artifacts are incomplete and the caller should
    run the experiment normally.  A result dict means the seed is complete and
    can be included in the aggregate without retraining it.
    """
    run_dir = Path(run_dir)
    eval_report = run_dir / "eval_report.json"

    if eval_report.is_file():
        try:
            with eval_report.open() as handle:
                result = json.load(handle)
            if result.get("Test"):
                logger.info("Reusing completed seed: %s", run_dir)
                return result
            logger.warning("Ignoring incomplete evaluation report: %s", eval_report)
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read evaluation report: %s", eval_report)

    required_artifacts = (
        "configuration.json",
        "report.json",
        "model.pt",
        "train_set.npy",
        "test_set.npy",
        "er_vocab.p",
        "re_vocab.p",
        "ee_vocab.p",
    )
    if not all((run_dir / filename).is_file() for filename in required_artifacts):
        return None

    logger.info("Training is complete; resuming evaluation only: %s", run_dir)
    try:
        from dicee.evaluation import Evaluator
        from dicee.static_funcs import load_json

        config = load_json(str(run_dir / "configuration.json"))
        config["full_storage_path"] = str(run_dir)
        args = SimpleNamespace(**config)
        model = _load_trained_model(run_dir)

        evaluator = Evaluator(args=args, is_continual_training=True)
        # The current resumed experiments all use NegSample, for which this
        # label is ignored.  EntityPrediction is also the correct fallback for
        # the entity-ranking configurations handled by these drivers.
        evaluator.dummy_eval(model, "EntityPrediction")
        result = dict(evaluator.report)
        if result.get("Test"):
            logger.info("Completed resumed evaluation: %s", run_dir)
            return result
        logger.warning("Resumed evaluation produced no Test metrics: %s", run_dir)
    except Exception:
        logger.exception("Could not resume evaluation; seed will be retrained: %s", run_dir)

    return None
