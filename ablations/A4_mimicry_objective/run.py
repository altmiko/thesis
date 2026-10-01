"""A4 - Mimicry objective: imitate the nearest Benign TRAIN flow instead of attacking the logits.

Design source: Traffic Manipulator (Han et al., IEEE JSAC 2021, Sec. VIII-B, Fig. 4), which
compares a search steered toward adversarial features with one that only imitates benign traffic
(a fixed target feature set).

``loss_mimicry`` replaces the victim-margin refinement loss with the squared distance, in
``asinh((x - center) / scale)`` space (the train-fit RobustScaler space with tails compressed),
between the relaxed adversarial flow and a fixed anchor: the nearest Benign flow of the TRAIN
split to the source flow (all Benign train flows searched; nothing from val/test). The anchor is
chosen once per source flow. The victim is still queried for the success predicate and the
incumbent (untargeted success = realized flow leaves its source class AND validator_v2), so the
arm measures whether victim-agnostic benign imitation inside the primitive box evades as often as
victim-guided search.

    python ablations/A4_mimicry_objective/run.py --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

from ablations.common.analysis import (  # noqa: E402
    append_report, success_profile, write_standard_report,
)
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.hybrid import HybridConfig  # noqa: E402
from ablations.common.runner import Condition  # noqa: E402

CONDITIONS = [
    Condition("loss_mimicry", "refinement minimizes distance to the nearest Benign train flow",
              HybridConfig(loss="mimicry")),
]


def analyze(results_dir: Path, reference_dir: Path) -> None:
    names = [c.name for c in CONDITIONS]
    write_standard_report(results_dir, reference_dir, names,
                          title="A4 - Mimicry objective vs victim-margin objective")
    append_report(results_dir, success_profile(results_dir, reference_dir, names))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__)
