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

def tuple_to_dict(file_name):
    print('Loading ', file_name)
    new_dict = dict()
    feature_list, label_list, path_list = np.load(file_name)
    assert(feature_list.shape == label_list.shape == path_list.shape)
    for triplet in zip(feature_list, label_list, path_list):
        x_data, y_data, z_data = triplet[0], triplet[1], triplet[2]
        if (y_data, z_data) in new_dict:
            new_dict[(y_data, z_data)].append(x_data)
        else:
            new_dict[(y_data, z_data)] = [x_data]
    return new_dict

def siw_protocol_01(train_dict, probe_dict, max_frames=60):
    new_probe_dict = dict()
    new_train_dict = dict()
    # Restrain number of samples per video for training
    for ((y_data, z_data), x_data) in train_dict.items():
        new_x_data = [feat for feat in x_data[0:max_frames]]
        train_dict[(y_data, z_data)] = new_x_data
    # Rename spoofing videos for binary classification
    for ((y_data, z_data), x_data) in probe_dict.items():
        if y_data != 'live':
            y_data = 'spoof'
        new_probe_dict[(y_data, z_data)] = x_data
    for ((y_data, z_data), x_data) in train_dict.items():
        if y_data != 'live':
            y_data = 'spoof'
        new_train_dict[(y_data, z_data)] = x_data
    return new_train_dict, new_probe_dict

def siw_protocol_02(train_dict, probe_dict):
    return train_dict, probe_dict

def siw_protocol_03(train_dict, probe_dict):
    return train_dict, probe_dict

def main():
    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC_curve.pdf', type=str)
    parser.add_argument('-e', '--error_outcome', help='Json', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-r', '--repetitions', help='Number of executions [10..INF]', required=False, default=1, type=int)
    parser.add_argument('-s', '--scenario', help='Choose protocol execution', required=False, default="one", type=str)
    parser.add_argument('-p', '--probe_file', help='Path to probe txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-probe-new.npy"), type=str)
    parser.add_argument('-t', '--train_file', help='Path to train txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-train-new.npy"), type=str)
    
    # Storing in variables
    args = parser.parse_args()
    CHART_PATH = str(args.chart_path)
    ERROR_OUTCOME = str(args.error_outcome)
    PROBE_FILE = str(args.probe_file)
    REPETITIONS = int(args.repetitions)
    SCENARIO = str(args.scenario)
    TRAIN_FILE = str(args.train_file)

    # Store all-interation results
    result_errors = dict()
    result_labels = list()
    result_scores = list()

    # Split dataset into train and test sets
    train_dict = tuple_to_dict(TRAIN_FILE)
    probe_dict = tuple_to_dict(PROBE_FILE)

    for index in range(REPETITIONS):
        print('> ITERATION ' + str(index + 1))
        c_train_dict, c_probe_dict = siw_protocol_01(train_dict, probe_dict, max_frames=60)

        # Instantiate SpoofDet class
        spoofDet = FaceSpoofing()
        spoofDet.import_features(feature_dict=c_train_dict)
        spoofDet.trainPLS(components=10, iterations=1000) 
        # spoofDet.trainSVM(kernel_type='linear', verbose=False)

        # Check whether class is ready to continue
        print('Classes: ', spoofDet.get_classes())
        assert('live' in spoofDet.get_classes())

        # Define APCER/BPCER variables
        instances = spoofDet.get_classes()
        counter_dict = {label:0.0 for label in instances}
        mistake_dict = {label:0.0 for label in instances}

        # Define lists to plot charts
        result = dict()
        result['labels'] = list()
        result['scores'] = list()

        # Predict samples
        video_counter = 0
        for (label, path) in c_probe_dict.keys():
            counter_dict[label] += 1
            scores = spoofDet.predict_feature(c_probe_dict[(label, path)])
            scores_dict = {label:value for (label,value) in scores}
            # Generate ROC Curve
            if len(scores_dict):
                if label == 'live':
                    result['labels'].append(+1)
                    result['scores'].append(scores_dict['live'])
                else:
                    result['labels'].append(-1)
                    result['scores'].append(scores_dict['live'])
                print(video_counter + 1, '>>', path, label, '>>', scores_dict)
            # Increment ERROR values
            if len(scores):
                pred_label, pred_score = scores[0]
                if pred_label != label:
                    mistake_dict[label] += 1
            # Increment counter
            video_counter += 1

        # Generate APCER, BPCER
        error_dict = {label:mistake_dict[label]/counter_dict[label] for label in instances}
        for label in error_dict.keys():
            if label in result_errors:
                result_errors[label].append(error_dict[label])
            else:
                result_errors[label] = [error_dict[label]]
        print("ERROR RESULT", error_dict)

        # Save data to files
        result_labels.append(result['labels'])
        result_scores.append(result['scores'])
        np.save('saves/data.npy', [result_errors, result_labels, result_scores])
        with open(ERROR_OUTCOME + '.json', 'w') as out_file:
            out_file.write(json.dumps(result_errors))

        # Plot figures
        plt.figure()
        roc_data = MyPlots.merge_roc_curves(result_labels, result_scores, name='ROC Average')
        MyPlots.plt_roc_curves([roc_data,])
        plt.savefig(CHART_PATH)
        plt.close()


if __name__ == "__main__":
    main()