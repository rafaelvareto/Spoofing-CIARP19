import argparse
import cv2 as cv
import json
import numpy as np
import os
import random
import time

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

def split_train_test_sets(complete_tuple_list, train_set_size=0.8):
    label_set = {label for (path, label) in complete_tuple_list}
    train_tuple = list()
    test_tuple = list()
    for label in label_set:
        # Sample images to compose train_set
        path_set = {path for (path, target) in complete_tuple_list if label == target}
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
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset"), type=str)
    parser.add_argument('-e', '--error_outcome', help='Json', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-r', '--repetitions', help='Number of executions [10..INF]', required=False, default=1, type=int)
    parser.add_argument('-te', '--testing_file', help='Path to testing txt file', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset/directions-2-test.txt"), type=str)
    parser.add_argument('-tr', '--training_file', help='Path to training txt file', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset/directions-2-train.txt"), type=str)
    
    # Storing in variables
    args = parser.parse_args()
    CHART_PATH = str(args.chart_path)
    ERROR_OUTCOME = str(args.error_outcome)
    FOLDER_PATH = str(args.folder_path)
    REPETITIONS = int(args.repetitions)
    TEST_FILE = str(args.testing_file)
    TRAIN_FILE = str(args.training_file)

    # Store all-interation results
    result_errors = dict()
    result_labels = list()
    result_scores = list()

    for index in range(REPETITIONS):
        print('> ITERATION ' + str(index + 1))

        # Split dataset into train and test sets
        test_set = load_txt_file(file_name=TEST_FILE)
        train_set = load_txt_file(file_name=TRAIN_FILE)

        # Instantiate SpoofDet class
        spoofDet = FaceSpoofing()
        spoofDet.obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=train_set, frame_drop=10, new_size=(400,300), verbose=True)
        spoofDet.trainEPLS(models=100, samples4model=50, pos_label='live', neg_label='spoofing', components=10, iterations=1000) 

        # Check whether class is ready to continue
        assert('live' in spoofDet.get_classes())

        # Define APCER/BPCER variables
        instances = spoofDet.get_classes()
        counter_dict = {label:0.0 for label in instances}
        mistake_dict = {label:0.0 for label in instances}
        
        # Define lists to plot charts
        result = dict()
        result['labels'] = list()
        result['scores'] = list()

        # TEST: Predict samples
        video_counter = 0
        for (path, label) in test_set:
            counter_dict[label] += 1
            probe_path = os.path.join(FOLDER_PATH, path)
            probe_video = cv.VideoCapture(probe_path)
            frame_count = probe_video.get(cv.CAP_PROP_FRAME_COUNT)
            start_time = time.time()
            pred_label, pred_score = spoofDet.predict_video(probe_video, threshold=0.5)
            finish_time = time.time()
            assert(pred_score is not None)
            # Generate ROC Curve
            result['labels'].append(+1) if label == 'live' else result['labels'].append(-1)
            result['scores'].append(pred_score)
            # Increment ERROR values
            if pred_label != label: mistake_dict[label] += 1
            # Increment counter
            video_counter += 1
            print(video_counter, '>>', label, '>', {pred_label:pred_score}, 'FPS:', frame_count / (finish_time - start_time))

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
    
