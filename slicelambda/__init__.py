"""slicelambda: how much slice-axis resolution a lesion ground truth hides.

Re-slicing of binary lesion masks, per-lesion Dice, the relation
R(T) = c + a exp(-lambda (T - T0) / d) and the run-length model.
"""
from .reslice import degrade, phases
from .lesions import CLASSES, depth_class, label, match, runs_of
from .experiment import run_mask, run_manifest, T_MNI, T_LESJAK
from .relation import Cohort, class_depth, predict, tail_depth
from .runs import estimate_lambda, lesion_predictions, measured_lambda

__version__ = "1.0.0"
