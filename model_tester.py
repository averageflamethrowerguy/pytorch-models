import torch
import torch.nn as nn
import datetime
import sys
sys.path.append('../')

from utilities import run
from utilities import Dataset
from utilities import data_loader
from utilities import test_correlation
from models import MLP

BATCH_SIZE=8184
number_conv_steps =32
LOOKBACK_DISTANCE=256
PREDICTION_RANGE=1
load_location="../benchmarks/finished_models/32_layer_resnet.pt"

train_generator, test_generator, NUM_FEATURES = data_loader.load_csv(
    '../Intraday-Stock-Data/a-f/A_2010_2019.txt', 
    LOOKBACK_DISTANCE, 
    PREDICTION_RANGE, 
    BATCH_SIZE, 
    torch.float32 
)

model = MLP.MLP(
    NUM_FEATURES=NUM_FEATURES,
    LOOKBACK_DISTANCE=LOOKBACK_DISTANCE,
    output_dim=1,
    number_conv_steps=number_conv_steps
)

model.load_state_dict(torch.load(load_location))

test_correlation.test_correlation(model, test_generator, NUMBER_ITERATIONS=1)
