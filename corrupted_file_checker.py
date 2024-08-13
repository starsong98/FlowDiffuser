"""
Checking if any of the files in FlyingThings dataset have been corrupted or something.

This sucks.
"""
from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import datasets
#import evaluate
#from flowdiffuser import FlowDiffuser

#val_dataset = datasets.FlyingThings3D(dstype='frames_cleanpass')
#val_dataset = datasets.FlyingThings3D(dstype='frames_finalpass')
#val_dataset = datasets.MpiSintel(split='training', dstype='clean')
val_dataset = datasets.FlyingChairs(split='training')

#out_filename = 'results/FT3D_cleanpass_checks.csv'
#out_filename = 'results/FT3D_finalpass_checks2.csv'
#out_filename = 'results/sintel_cleanpass_checks.csv'
out_filename = 'results/chairs_train_checks.csv'

lines_to_save = [['filename0', 'image0 size', 'filename1', 'image1 size', 'flow file', 'flow map size', 'nan ratio']]
for val_id in tqdm(range(len(val_dataset))):
    image0, image1, flow_gt, _ = val_dataset[val_id]
    filename0 = val_dataset.image_list[val_id][0]
    filename1 = val_dataset.image_list[val_id][1]
    flowname = val_dataset.flow_list[val_id]
    nan_count = torch.isnan(flow_gt).sum().item()
    if nan_count > 0:
        print(f"NaN count on val id {val_id}: {nan_count}")
    lines_to_save.append([filename0, image0.shape, filename1, image1.shape, flowname, flow_gt.shape, nan_count])
        


with open(out_filename, 'a+', newline="") as fp:
    writer = csv.writer(fp)
    writer.writerows(lines_to_save)