# Experiment Launching file

import argparse
import os.path
import numpy as np
import torch
from train import train

if __name__ == '__main__':
    train('mnist', debug=False)
