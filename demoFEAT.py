import argparse
import cv2 as cv
import json
import numpy as np
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from face_spoofing import FaceSpoofing
from myPlots import MyPlots
from video import Video

HOME = os.path.expanduser("~")

def tuple_to_dict(file_name, max_frames=None):
    print('Loading ', file_name)
    new_dict = dict()
    feature_list, label_list, path_list = np.load(file_name)
    print(feature_list.shape, label_list.shape, path_list.shape)
    for triplet in zip(feature_list, label_list, path_list):
        x_data, y_data, z_data = triplet[0], triplet[1], triplet[2]
        if (z_data in new_dict) and (max_frames is None):
            new_dict[z_data].append(x_data)
        elif (z_data in new_dict) and (max_frames is not None):
            if len(new_dict[z_data]) < max_frames:
                new_dict[z_data].append(x_data)
        else:
            new_dict[z_data] = [x_data]
    return new_dict

def load_txt_file(file_name):
    this_file = open(file_name, 'r')
    this_list = list()
    for line in this_file:
        line = line.rstrip()
        components = line.split()
        this_list.append(components)
    return this_list

def split_train_test_sets(tuple_list, train_set_size=0.8):
    label_set = {label for (path, label) in tuple_list}
    train_tuple = list()
    test_tuple = list()
    for label in label_set:
        # Sample images to compose train_set
        path_set = {path for (path, target) in tuple_list if label == target}
        train_set = set(random.sample(path_set, int(train_set_size * len(path_set))))
        test_set = path_set - train_set
        # Put together labels and paths
        train_tuple.extend([(path, label) for path in train_set])
        test_tuple.extend([(path, label) for path in test_set])
    return train_tuple, test_tuple

def main():
    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC_curve.pdf', type=str)
    parser.add_argument('-e', '--error_outcome', help='Json', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-p', '--probe_file', help='Path to probe txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-probe-50p-new.npy"), type=str)
    parser.add_argument('-t', '--train_file', help='Path to train txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-train-50p-new.npy"), type=str)
    
    # Storing in variables
    args = parser.parse_args()
    CHART_PATH = str(args.chart_path)
    ERROR_OUTCOME = str(args.error_outcome)
    PROBE_FILE = str(args.probe_file)
    TRAIN_FILE = str(args.train_file)

    # Store all-interation results
    result_errors = dict()
    result_labels = list()
    result_scores = list()

    # Split dataset into train and test sets
    train_dict = tuple_to_dict(TRAIN_FILE, max_frames=60)
    probe_dict = tuple_to_dict(PROBE_FILE, max_frames=None)

    # Instantiate SpoofDet class
    spoofDet = FaceSpoofing()
    spoofDet.load_features(file_name='saves/protocol_01_train.npy_new.npy', new_size=(400,300))
    spoofDet.trainPLS(components=10, iterations=1000) 
    # spoofDet.trainSVM(kernel_type='linear', verbose=False)

    # Check whether class is ready to continue
    assert('live' in spoofDet.get_classes())


if __name__ == "__main__":
    main()