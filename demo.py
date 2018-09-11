import argparse
import cv2 as cv
import numpy as np
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from face_spoofing import FaceSpoofing
from myPlots import MyPlots
from video import Video

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
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='ROC_curve.pdf', type=str)
    parser.add_argument('-d', '--direction_path', help='Path to video txt file', required=False, default='datasets/SiW-dataset/directions-3-small.txt', type=str)
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default='datasets/SiW-dataset/', type=str)
    parser.add_argument('-r', '--repetitions', help='Number of executions [10..INF]', required=False, default=10, type=int)
    parser.add_argument('-t', '--train_set_size', help='Dataset percentage comprising training set [0..1]', required=False, default=0.5, type=float)
    
    # Storing in variables
    args = parser.parse_args()
    CHART_PATH = str(args.chart_path)
    DETECT = False
    DIRECT_PATH = str(args.direction_path)
    FOLDER_PATH = str(args.folder_path)
    REPETITIONS = int(args.repetitions)
    TRAIN_SIZE = float(args.train_set_size)

    # Store all-interation results
    result_labels = list()
    result_scores = list()

    for index in range(REPETITIONS):
        print('> ITERATION ' + str(index + 1))

        # Split dataset into train and test sets
        complete_set = load_txt_file(file_name=DIRECT_PATH)
        train_set, test_set = split_train_test_sets(complete_tuple_list=complete_set, train_set_size=TRAIN_SIZE)

        # Instantiate SpoofDet class
        spoofDet = FaceSpoofing()
        spoofDet.obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=train_set, detect=DETECT, verbose=True)
        spoofDet.trainPLS(components=10, iterations=1000) # spoofDet.trainSVM(kernel_type='linear', verbose=False)
        spoofDet.save_model()

        # Check whether class is ready to continue
        # assert(len(spoofDet.get_classes()) == 2)
        assert('live' in spoofDet.get_classes())
        # assert('spoofing' in spoofDet.get_classes())

        # Define APCER/BPCER variables
        instances = spoofDet.get_classes()
        counter_dict = {label:0.0 for label in instances}
        mistake_dict = {label:0.0 for label in instances}
        
        # Define lists to plot charts
        result = dict()
        result['labels'] = list()
        result['scores'] = list()
        
        # Predict samples
        for (path, label) in test_set:
            print('>> ', path, label)
            counter_dict[label] += 1
            probe_path = os.path.join(FOLDER_PATH, path)
            probe_video = cv.VideoCapture(probe_path)
            scores = spoofDet.predict_video(probe_video, detect=DETECT)
            scores_dict = {label:value for (label,value) in scores}
            
            # Generate ROC Curve
            if len(scores_dict):
                if label == 'live':
                    result['labels'].append(+1)
                    result['scores'].append(scores_dict['live'])
                else:
                    result['labels'].append(-1)
                    result['scores'].append(scores_dict['live'])
                print(scores_dict)

            # Increment ERROR values
            pred_label, pred_score = scores[0]
            if pred_label != label:
                mistake_dict[label] += 1

        # Generate APCER, BPCER
        result_dict = {label:mistake_dict[label]/counter_dict[label] for label in instances}
        print("RESULT", counter_dict, mistake_dict, result_dict)

        # Save data to files
        result_labels.append(result['labels'])
        result_scores.append(result['scores'])
        np.save('data.npy', [result_labels, result_scores])

        # Plot figures
        plt.figure()
        roc_data = MyPlots.merge_roc_curves(result_labels, result_scores, name='ROC Average')
        MyPlots.plt_roc_curves([roc_data,])
        plt.savefig(CHART_PATH)
        plt.close()   
                
    # current_video = Video()
    # current_video.set_input_video(VIDEO_PATH)
    # current_video.play(spoofer=spoofDet, delay=1)

if __name__ == "__main__":
    main()
    