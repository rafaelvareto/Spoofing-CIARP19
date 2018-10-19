import argparse
import cv2 as cv
import json
import numpy as np
import os
import random
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc

from face_spoofing import FaceSpoofing
from myPlots import MyPlots
from video import Video

HOME = os.path.expanduser("~")


def binarize_label(train_dict, probe_dict, input_label='live', pos_label='live', neg_label='spoof'):
    '''
    Rename spoofing videos for binary classification
    ''' 
    new_probe_dict = dict()
    new_train_dict = dict()
    for ((y_data, z_data), x_data) in probe_dict.items():
        if y_data != input_label:
            y_data = neg_label
        else:
            y_data = pos_label
        new_probe_dict[(y_data, z_data)] = x_data
    for ((y_data, z_data), x_data) in train_dict.items():
        if y_data != input_label:
            y_data = neg_label
        else:
            y_data = pos_label
        new_train_dict[(y_data, z_data)] = x_data
    return new_train_dict, new_probe_dict

def drop_frames(sample_dict, skip_frames=10):
    '''
    Skip some video frames in order to avoid overfitting 
    '''
    for ((y_data, z_data), x_data) in sample_dict.items():
        new_x_data = [feat for (idx, feat) in enumerate(x_data) if idx % skip_frames == 0]
        sample_dict[(y_data, z_data)] = new_x_data
    return sample_dict

def limit_frames(sample_dict, max_frames=60):
    '''
    Restrain number of samples per video
    '''
    for ((y_data, z_data), x_data) in sample_dict.items():
        new_x_data = [feat for feat in x_data[0:max_frames]]
        sample_dict[(y_data, z_data)] = new_x_data
    return sample_dict

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

def tokenize_path(path_name):
    file_name = os.path.basename(path_name)
    tokens = file_name.split('-') 
    numbers = [re.sub('[^0-9]', '', token) for token in tokens]
    return [int(number) for number in numbers]

def tuple_to_dict(file_name, binarize=False):
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
    '''
    Set maximum number of frames per video for training samples
    '''
    for ((y_data, z_data), x_data) in train_dict.items():
        new_x_data = [feat for feat in x_data[0:max_frames]]
        train_dict[(y_data, z_data)] = new_x_data
    return train_dict, probe_dict

def siw_protocol_02(train_dict, probe_dict, medium_out=1, max_frames=False, skip_frames=False):
    '''
    Filter out media types that do not satisfy protocol two (replay attack) by keeping a single replay attack medium out at a time.
    File name information: SubjectID_SensorID_TypeID_MediumID_SessionID.mov
    '''
    new_probe_dict = dict()
    new_train_dict = dict()
    for ((y_data, z_data), x_data) in probe_dict.items():
        subject, sensor, category, medium, session = tokenize_path(z_data)
        if (category == 1):
            new_probe_dict[(y_data, z_data)] = x_data
        elif (category == 3) and (medium == medium_out):
            new_probe_dict[(y_data, z_data)] = x_data
    for ((y_data, z_data), x_data) in train_dict.items():
        subject, sensor, category, medium, session = tokenize_path(z_data)
        if (category == 1):
            new_train_dict[(y_data, z_data)] = x_data
        elif ((category == 3) or (category == 3)) and (medium != medium_out):
            new_train_dict[(y_data, z_data)] = x_data
    if skip_frames:
        new_train_dict = drop_frames(new_train_dict, skip_frames=skip_frames)
    if max_frames:
        new_train_dict = limit_frames(new_train_dict, max_frames=max_frames)
    return new_train_dict, new_probe_dict

def siw_protocol_03(train_dict, probe_dict, category_out=2, max_frames=False, skip_frames=False):
    '''
    Filter out media types that do not satisfy protocol three by performing a person attack testing from print to replay attack and vice-versa.
    File name information: SubjectID_SensorID_TypeID_MediumID_SessionID.mov
    '''
    new_probe_dict = dict()
    new_train_dict = dict()
    for ((y_data, z_data), x_data) in probe_dict.items():
        subject, sensor, category, medium, session = tokenize_path(z_data)
        if (category == 1) or (category == category_out):
            new_probe_dict[(y_data, z_data)] = x_data
    for ((y_data, z_data), x_data) in train_dict.items():
        subject, sensor, category, medium, session = tokenize_path(z_data)
        if (category == 1) or (category != category_out):
            new_train_dict[(y_data, z_data)] = x_data
    if skip_frames:
        new_train_dict = drop_frames(new_train_dict, skip_frames=skip_frames)
    if max_frames:
        new_train_dict = limit_frames(new_train_dict, max_frames=max_frames)
    return new_train_dict, new_probe_dict

def set_aside_validation(dictionary, percent=0.10):
    keys_list = dictionary.keys()
    num_samples = int(percent * len(keys_list))
    keys_rand = random.sample(keys_list, num_samples)
    keys_left = set(keys_list) - set(keys_rand)
    valid_dictionary = {key:dictionary[key] for key in keys_rand}
    prime_dictionary = {key:dictionary[key] for key in keys_left}
    return prime_dictionary, valid_dictionary

def main():
    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-a', '--aside', help='Set percentage of dataset to be used for threshold estimation', required=False, default=False, type=float)
    parser.add_argument('-b', '--bagging', help='Determine whether to run single or bassing-based approach', required=False, default=False, type=int)
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC_curve.pdf', type=str)
    parser.add_argument('-d', '--drop_frames', help='Skip some frames for training', required=False, default=False, type=int)
    parser.add_argument('-e', '--error_outcome', help='Json containing output APCER and BPCER', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-i', '--instances', help='Number of samples per bagging model', required=False, default=50, type=int)
    parser.add_argument('-m', '--max_frames', help='Establish maximum number of frames for training', required=False, default=False, type=int)
    parser.add_argument('-s', '--scenario', help='Choose protocol execution', required=False, default='one', type=str)
    parser.add_argument('-p', '--probe_file', help='Path to probe txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-probe.npy"), type=str)
    parser.add_argument('-t', '--train_file', help='Path to train txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-train.npy"), type=str)
    parser.add_argument('-th', '--threshold', help='Set threshold for probe prediction', required=False, default=0.0, type=float)
    
    # Storing in variables
    args = parser.parse_args()
    ASIDE = float(args.aside)
    BAGGING = int(args.bagging)
    CHART_PATH = str(args.chart_path)
    DROP_FRAMES = int(args.drop_frames)
    ERROR_OUTCOME = str(args.error_outcome)
    INSTANCES = int(args.instances)
    MAX_FRAMES = int(args.max_frames)
    SCENARIO = str(args.scenario)
    PROBE_FILE = str(args.probe_file)
    TRAIN_FILE = str(args.train_file)

    # Determining number of iterations
    if SCENARIO == 'one':
        REPETITIONS = 1
    elif SCENARIO == 'two':
        REPETITIONS = 4
    elif SCENARIO == 'three':
        REPETITIONS = 2
    else:
        raise ValueError('ERROR: Scenarios range from one through three.')
        exit

    # Store all-interation results
    max_neg_values = list()
    min_pos_values = list()
    result_errors = dict()
    result_labels = list()
    result_scores = list()
    thresholds = dict()

    # Split dataset into train and test sets
    train_dict = tuple_to_dict(TRAIN_FILE)
    probe_dict = tuple_to_dict(PROBE_FILE)

    for index in range(REPETITIONS):
        print('> ITERATION ' + str(index + 1))
        max_neg_value = -10.0
        min_pos_value = +10.0

        if SCENARIO == 'one':
            c_train_dict, c_probe_dict = siw_protocol_01(train_dict, probe_dict, max_frames=60)
        elif SCENARIO == 'two':
            c_train_dict, c_probe_dict = siw_protocol_02(train_dict, probe_dict, medium_out=index+1, max_frames=MAX_FRAMES, skip_frames=DROP_FRAMES)
            c_train_dict, c_valid_dict = set_aside_validation(c_train_dict, percent=ASIDE)
        elif SCENARIO == 'three':
            c_train_dict, c_probe_dict = siw_protocol_03(train_dict, probe_dict, category_out=index+2, max_frames=MAX_FRAMES, skip_frames=DROP_FRAMES)

        # Change into a binary problem
        c_train_dict, c_probe_dict = binarize_label(c_train_dict, c_probe_dict, input_label='live', pos_label='live', neg_label='spoof')

        # Print data size
        print('Train size:', len(c_train_dict), sum([1 for (y_data, z_data) in c_train_dict.keys() if y_data == 'live']), sum([1 for (y_data, z_data) in c_train_dict.keys() if y_data != 'live']))
        print('Probe size:', len(c_probe_dict), sum([1 for (y_data, z_data) in c_probe_dict.keys() if y_data == 'live']), sum([1 for (y_data, z_data) in c_probe_dict.keys() if y_data != 'live']))

        # Instantiate SpoofDet class
        spoofDet = FaceSpoofing()
        spoofDet.import_features(feature_dict=c_train_dict)
        if BAGGING:
            spoofDet.trainEPLS(models=BAGGING, samples4model=INSTANCES, components=10, iterations=1000) 
        else:
            spoofDet.trainPLS(components=10, iterations=1000)

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

        # THRESHOLD: Predict samples
        validation_labels = list()
        validation_scores = list()
        for (label, path) in c_valid_dict.keys():
            pred_label, pred_score = spoofDet.predict_feature(c_valid_dict[(label, path)])
            validation_labels.append(+1) if label == 'live' else validation_labels.append(-1)
            validation_scores.append(pred_score)
        precision, recall, threshold = precision_recall_curve(validation_labels, validation_scores)
        fmeasure = [(thr, (2 * (pre * rec) / (pre + rec))) for pre, rec, thr in zip(precision[:-1], recall[:-1], threshold)]
        fmeasure.sort(key=lambda tup:tup[1], reverse=True)
        best_threshold = fmeasure[0][0]
        print('SELECTED THRESHOLD', best_threshold)

        # TEST: Predict samples
        video_counter = 0
        for (label, path) in c_probe_dict.keys():
            counter_dict[label] += 1
            pred_label, pred_score = spoofDet.predict_feature(c_probe_dict[(label, path)], threshold=best_threshold)
            assert(pred_score is not None)
            # Generate ROC Curve
            result['labels'].append(+1) if label == 'live' else result['labels'].append(-1)
            result['scores'].append(pred_score)
            # Update MIN and MAX values
            if pred_score < min_pos_value and label == 'live': min_pos_value = pred_score 
            if pred_score > max_neg_value and label != 'live': max_neg_value = pred_score
            # Increment ERROR values
            if pred_label != label: mistake_dict[label] += 1
            # Increment counter
            video_counter += 1
            print(video_counter, '>>', path, label, '>>', {pred_label:pred_score}, counter_dict)

        # Generate APCER, BPCER
        error_dict = {label:mistake_dict[label]/counter_dict[label] for label in instances}
        for label in error_dict.keys():
            if label in result_errors:
                result_errors[label].append(error_dict[label])
            else:
                result_errors[label] = [error_dict[label]]
        max_neg_values.append(max_neg_value)
        min_pos_values.append(min_pos_value)
        print("ERROR RESULT", error_dict, 'max_neg_value:', max_neg_value, 'min_pos_value:', min_pos_value)

        # Generate F-Measure and find best threshold
        precision, recall, threshold = precision_recall_curve(result['labels'], result['scores'])
        fscores = [(thr, (2 * (pre * rec) / (pre + rec))) for pre, rec, thr in zip(precision[:-1], recall[:-1], threshold)]
        fscores.sort(key=lambda tup:tup[1], reverse=True)
        thresholds[index] = fscores[0]

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

    # Compute average APCER and BPCER
    print('\n------------------------------------------------------------------')
    print('(THRESHOLDS, ROC ACCURACY)', thresholds)
    print('\nMAX_NEG_VALUES:', max_neg_values, 'MIN_POS_VALUES:', min_pos_values)
    print("ERROR RESULT per ITERATION:", result_errors)
    for label in result_errors.keys():
        error_avg = np.mean(result_errors[label])
        error_std = np.std(result_errors[label])
        print("ERROR RESULT (label, avg, std):", label, error_avg, error_std)


if __name__ == "__main__":
    main()
