import torch
import numpy as np
import sys
sys.path.append(r"d:\thesis\src")
import run_attacks
import traceback

try:
    run_attacks.main(use_train_bounds=True, use_ste=True)
except Exception as e:
    traceback.print_exc()
