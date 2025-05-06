"""This module evaluates ABX discrimination tasks using the zrc_abx2 library and saves the results to JSON files."""

import json
from pathlib import Path

import zrc_abx2


def evaluate_and_save_abx(
    item_file_path: str,
    features_dir: str,
    output_path: str,
    context_mode: str = "within",
    speaker_mode: str = "within",
    feature_rate: float = 50.0,
    file_extension: str = ".pt",
    distance_mode: str = "cosine",
    max_size_group: int = 10,
    max_x_across: int = 5,
    seed: int = 3459,
    cuda: bool = False,
):
    """Evaluate ABX discrimination and save the results as a JSON file.

    Args:
        item_file_path (Path): Path to the ABX item file specifying test triplets.
        features_dir (Path): Directory containing feature (.npy) files for evaluation.
        output_path (Path): Path to write the evaluation JSON results.
        context_mode (str): Contextual grouping mode for ABX ('within' or 'any').
        speaker_mode (str): Speaker grouping mode for ABX ('within', 'across', etc).
    """
    features_dir = Path(features_dir)
    item_file_path = Path(item_file_path)
    output_path = Path(output_path)

    # fmt: off
    assert item_file_path.exists(), f"Item file {item_file_path} does not exist"
    assert features_dir.exists(), f"Features directory {features_dir} does not exist"
    assert features_dir.is_dir(), f"Features directory {features_dir} is not a directory"
    assert not output_path.exists(), f"Output file {output_path} already exists"
    # fmt: on

    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = zrc_abx2.EvalArgs(
        path_data=str(features_dir),
        path_item_file=str(item_file_path),
        file_extension=file_extension,
        feature_size=1 / feature_rate,
        cuda=cuda,
        context_mode=context_mode,
        speaker_mode=speaker_mode,
        distance_mode=distance_mode,
        max_size_group=max_size_group,
        max_x_across=max_x_across,
        seed=seed,
    )

    result = zrc_abx2.EvalABX().eval_abx(args)[0]

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)
