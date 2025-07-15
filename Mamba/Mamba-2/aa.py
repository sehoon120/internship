import torch
import scipy.linalg
import os
import math


# Hadamard like matrix로 바꿔서 다시만들어보기


current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
model_path = os.path.join(base_dir, "mamba2-130m")
print(model_path)