# -*- coding: utf-8 -*-
"""face_spoofing_swax.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MgFPK3VdGVRrXc5KdGOHUfosh5AC1vXq
"""


"""# Face-Spoofing MLP (SWAX)"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %tensorflow_version 2.x

import argparse
import cv2 as cv
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re

# from google.colab import drive
from joblib import Parallel, delayed
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.keras.models import Sequential as keras_sequential
from tensorflow.keras.layers import BatchNormalization as keras_batch_norm
from tensorflow.keras.layers import Dense as keras_dense 
from tensorflow.keras.layers import Dropout as keras_dropout
from tensorflow.keras import utils as keras_np_utils


DESC_VECTOR = ['hog', 'lbp1', 'lbp2', 'lbp', 'glcm', 'combined']

ASIDE = False
BAGGING = False #100
DESCRIPTOR = DESC_VECTOR[5]
DROP_FRAMES = False
INSTANCES = 500
MAX_FRAMES = False
SCENARIO = 'one'

DATASET_FILE = 'datasets/SWAX-dataset.npy'


"""## GENERAL FUNCTIONS"""

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

def set_aside_validation(dictionary, percent=0.10):
    keys_list = dictionary.keys()
    num_samples = int(percent * len(keys_list))
    keys_rand = random.sample(keys_list, num_samples)
    keys_left = set(keys_list) - set(keys_rand)
    valid_dictionary = {key:dictionary[key] for key in keys_rand}
    prime_dictionary = {key:dictionary[key] for key in keys_left}
    return prime_dictionary, valid_dictionary

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

def trim_descriptor_data(dictionary, beg_index, end_index):
    '''
    OULU (hog, lbp-1, lbp-2, glcm, total): 256 795 795 48 1894
    SWAX (hog, lbp-1, lbp-2, glcm, total):  32 795 795 48 1670
    '''
    new_dict = dict()
    for (key, values) in dictionary.items():
        new_values = [value[beg_index:end_index] for value in values]
        new_dict[key] = new_values
    print('Feature Size:', len(values[0]), len(new_values[0]))
    return new_dict

def tuple_to_dict(file_name, binarize=False):
    print('Loading ', file_name)
    new_dict = dict()
    feature_list, label_list, path_list = np.load(file_name, allow_pickle=True, encoding="latin1")
    assert(feature_list.shape == label_list.shape == path_list.shape)
    label_set = set(label_list)
    print(label_set)
    for triplet in zip(feature_list, label_list, path_list):
        x_data, y_data, z_data = triplet[0], triplet[1], triplet[2]
        if (y_data, z_data) in new_dict:
            new_dict[(y_data, z_data)].append(x_data)
        else:
            new_dict[(y_data, z_data)] = [x_data]
    return new_dict
    

"""## Dataset functions"""

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
        # input()
    return new_train_dict, new_probe_dict

def swax_tokenize_path(path_name):
    file_name = os.path.basename(path_name)
    tokens = file_name.split('-') 
    numbers = [re.sub('[^0-9]', '', token) for token in tokens]
    return [int(number) for number in numbers]

def swax_protocol_01(dataset_dict, curr_label, max_frames=60):
    new_probe_dict = dict()
    new_train_dict = dict()
    '''
    Protocol 01: Unsupervised, with additional data
    - Procedures claimed to be unsupervised cannot make use of presentation attack samples at the training stage, being restricted to authentic samples only.
    - There should be no parameters carrying bona fide or counterfeit labels, not to mention any other relevant information such as file names or singular identifiers.
    - Approaches can neither benefit from “beforehand information” concerning the number of samples included in each class nor use the label distribution inherent to training and test sets.
    - Supplementary samples, outside of SWAX database, are allowed in cases where they depict bona fide individuals only and should not be composed of hand-labeled data or information indicating whether pictures are authentic or comprise presentation attacks.
    '''
    for ((y_data, z_data), x_data) in dataset_dict.items():
        if (curr_label not in z_data) and ('real' in z_data):
            new_train_dict[(y_data, z_data)] = x_data
        elif (curr_label in z_data):
            new_probe_dict[(y_data, z_data)] = x_data
    return new_train_dict, new_probe_dict

def swax_protocol_02(train_dict, probe_dict, medium_out=1, max_frames=False, skip_frames=False):
    '''
    Filter out media types that do not satisfy protocol two (replay attack) by keeping a single replay attack medium out at a time.
    File name information: SubjectID_SensorID_TypeID_MediumID_SessionID.mov
    '''
    new_probe_dict = dict()
    new_train_dict = dict()
    for ((y_data, z_data), x_data) in probe_dict.items():
        subject, sensor, category, medium, session = swax_tokenize_path(z_data)
        if (category == 1):
            new_probe_dict[(y_data, z_data)] = x_data
        elif (category == 3) and (medium == medium_out):
            new_probe_dict[(y_data, z_data)] = x_data
    for ((y_data, z_data), x_data) in train_dict.items():
        subject, sensor, category, medium, session = swax_tokenize_path(z_data)
        if (category == 1):
            new_train_dict[(y_data, z_data)] = x_data
        elif ((category == 3) or (category == 3)) and (medium != medium_out):
            new_train_dict[(y_data, z_data)] = x_data
    if skip_frames:
        new_train_dict = drop_frames(new_train_dict, skip_frames=skip_frames)
    if max_frames:
        new_train_dict = limit_frames(new_train_dict, max_frames=max_frames)
    return new_train_dict, new_probe_dict

def swax_protocol_03(train_dict, probe_dict, category_out=2, max_frames=False, skip_frames=False):
    '''
    Filter out media types that do not satisfy protocol three by performing a person attack testing from print to replay attack and vice-versa.
    File name information: SubjectID_SensorID_TypeID_MediumID_SessionID.mov
    '''
    new_probe_dict = dict()
    new_train_dict = dict()
    for ((y_data, z_data), x_data) in probe_dict.items():
        subject, sensor, category, medium, session = swax_tokenize_path(z_data)
        if (category == 1) or (category == category_out):
            new_probe_dict[(y_data, z_data)] = x_data
    for ((y_data, z_data), x_data) in train_dict.items():
        subject, sensor, category, medium, session = swax_tokenize_path(z_data)
        if (category == 1) or (category != category_out):
            new_train_dict[(y_data, z_data)] = x_data
    if skip_frames:
        new_train_dict = drop_frames(new_train_dict, skip_frames=skip_frames)
    if max_frames:
        new_train_dict = limit_frames(new_train_dict, max_frames=max_frames)
    return new_train_dict, new_probe_dict


"""## Face Spoofing Class"""

class FaceSpoofing:
    def __init__(self):
        print('Face Spoofing Class')
        self._color_space = 0
        self._dictionary = dict()
        self._gaussian_var = 2.0
        self._features = list()
        self._fourier = list()
        self._images = list()
        self._kernel_size = 7
        self._labels = list()
        self._neg_label = 'None'
        self._models = None
        self._paths = list()
        self._pos_label = 'None'
        self._size = (640, 360)
        self._type = 'None'
        self._vr_height = 1
        self._vr_width = 30

    def __build_dictionary(self):
        lab_dict = {label:number for (number,label) in zip(range(self.get_num_classes()), self.get_classes())}
        num_dict = {number:label for (number,label) in zip(range(self.get_num_classes()), self.get_classes())}
        self._dictionary = dict( list(lab_dict.items()) + list(num_dict.items()) )
        print(self._dictionary)

    def __channel_swap(self, image): 
        spare = copy.copy(image)
        image[:, :, 0] = spare[:, :, 2]
        image[:, :, 2] = spare[:, :, 0]
        return image

    def __feature_sampling(self, num_samples=100):
        rand_features = list()
        rand_labels = list()
        for cat in self.get_classes():
            cat_indices = [index for (index,value) in enumerate(self._labels) if value == cat]
            cat_sampled = random.sample(cat_indices, num_samples)
            cat_features = [self._features[index] for index in cat_sampled]
            cat_labels = [self._labels[index] for index in cat_sampled]
            rand_features.extend(cat_features)
            rand_labels.extend(cat_labels)
        return rand_features, rand_labels

    def __manage_bagging_results(self, dictionary, neg_list, pos_list):
        pos_value = sum(pos_list)/len(pos_list)
        neg_value = sum(neg_list)/len(neg_list)
        if self._pos_label in dictionary:
            dictionary[self._pos_label].append(pos_value)
        else:
            dictionary[self._pos_label] = [pos_value]
        if self._neg_label in dictionary:
            dictionary[self._neg_label].append(neg_value)
        else:
            dictionary[self._neg_label] = [neg_value]
        return dictionary

    def __manage_results(self, dictionary, score_list):
        for (label, result) in score_list:
            if label in dictionary:
                dictionary[label].append(result)
            else:
                dictionary[label] = [result]
        for label in self.get_classes():
            if label not in dictionary:
                dictionary[label] = [0.0]
        return dictionary

    def __mean_and_return(self, dictionary):
        new_list = {key:float(np.mean(value)) for (key, value) in dictionary.items()}
        return new_list

    def __mean_and_sort(self, dictionary):
        new_list = [(key,float(np.mean(value))) for (key, value) in dictionary.items()]
        new_list.sort(key=lambda tup:tup[1], reverse=True)
        return new_list

    def get_gray_image(self, color_img):
        return cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    def get_residual_noise(self, gray_img, filter_type='median'):
        if filter_type == 'median':
            blurred = cv.medianBlur(src=gray_img, ksize=self._kernel_size)
            noise = cv.subtract(gray_img, blurred)
        elif filter_type == 'gaussian':
            blurred = cv.GaussianBlur(src=gray_img, ksize=(self._kernel_size, self._kernel_size), sigmaX=self._gaussian_var, sigmaY=0.0)
            noise = cv.subtract(gray_img, blurred)
        else:
            raise ValueError('ERROR: Two smoothing methods available: median and gaussian')
        return noise

    def get_classes(self):
        categories = set(self._labels)
        return list(categories)

    def get_num_classes(self):
        categories = set(self._labels)
        return len(categories)

    def import_features(self, feature_dict):
        for ((label, path), features) in feature_dict.items():
            # print('Imported Features: ', (label, path), len(features), len(features[0]))
            for feat in features:
                self._features.append(np.asarray(feat))
                self._labels.append(label)
                self._paths.append(path)

    def load_model(self, file_name='saves/model.npy'):
        self._labels, self._models, self._type = np.load(file_name)

    def predict_feature_oaa(self, probe_features, drop_frame, threshold):
        class_dict = dict()
        for (index, feature) in enumerate(probe_features):
            if index % drop_frame == 0:
                results = [float(model[0].predict(np.array([feature]))) for model in self._models]
                labels = [model[1] for model in self._models]
                scores = list(map(lambda left,right:(left,right), labels, results))
                class_dict = self.__manage_results(class_dict, scores)
        class_list = self.__mean_and_sort(class_dict)
        best_label, best_score = class_list[0]
        return (best_label, best_score)

    def predict_feature_ocsvm(self, probe_features, threshold):
        binnary_list = list()
        for model in self._models:
            results = model.predict(np.asarray(probe_features))
            binnary = [+1.0 if result > 0.0 else 0.0 for result in results]
            binnary_list.append(binnary)
        ratio_list = [sum(array) / len(array) for array in binnary_list]
        score = np.mean(ratio_list)
        label = self._pos_label if score >= threshold else self._neg_label
        return (label, score)

    def predict_feature_bag(self, probe_features, threshold):
        binnary_list = list()
        for model in self._models:
            results = model.predict(np.asarray(probe_features))
            binnary = [+1.0 if result > 0.1 else 0.0 for result in results]
            binnary_list.append(binnary)
        binnary_list = np.transpose(binnary_list)
        ratio_list = [sum(array) / len(array) for array in binnary_list]
        score = np.mean(ratio_list)
        label = self._pos_label if score >= threshold else self._neg_label
        return (label, score)

    def predict_feature_mlp(self, probe_features, threshold):
        binnary_list = list()
        for model in self._models:
            results = model.predict(np.asarray(probe_features))
            binnary = [+1.0 if result.flatten()[1] > 0.6 else 0.0 for result in results]
            binnary_list.append(binnary)
        binnary_list = np.transpose(binnary_list)
        ratio_list = [sum(array) / len(array) for array in binnary_list]
        score = np.mean(ratio_list)
        label = self._pos_label if score >= threshold else self._neg_label
        return (label, score)

    def predict_feature(self, probe_features, drop_frame=2, threshold=0.50):
        if self._type in ['OAAPLS', 'OAASVM']:
            return self.predict_feature_oaa(probe_features, drop_frame, threshold)
        elif self._type in ['OCSVM']:
            return self.predict_feature_ocsvm(probe_features, threshold)
        elif self._type in ['EPLS', 'ESVM']:
            return self.predict_feature_bag(probe_features, threshold)
        elif self._type in ['EMLP']:
            return self.predict_feature_mlp(probe_features, threshold)
        else:
            return (None, None, None)

    def predict_image(self, probe_image):
        if self._type in ['OAAPLS', 'OAASVM']:
            class_dict = dict()
            scaled_image = cv.resize(probe_image, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
            feature = self.gray2feat_pipeline(scaled_image)
            results = [float(model[0].predict(np.array([feature]))) for model in self._models]
            labels = [model[1] for model in self._models]
            scores = list(map(lambda left,right:(left,right), labels, results))
            class_dict = self.__manage_results(class_dict, scores)
            return self.__mean_and_sort(class_dict)
        else:
            raise ValueError('Error predicting probe image') 

    def predict_video(self, probe_video, drop_frame=10, threshold=0.50):
        frame_counter = 0
        class_dict = dict()
        probe_features = list()
        while(probe_video.isOpened()):
            ret, probe_frame = probe_video.read()
            if ret:
                if frame_counter % drop_frame == 0:
                    scaled_frame = cv.resize(probe_frame, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
                    feature = self.gray2feat_pipeline(scaled_frame)
                    probe_features.append(feature)
                frame_counter += 1
            else: break
        if self._type in ['OAAPLS', 'OAASVM']:
            return self.predict_feature_oaa(probe_features, drop_frame=1, threshold=threshold)
        elif self._type in ['EPLS', 'ESVM']:
            return self.predict_feature_bag(probe_features, drop_frame=1, threshold=threshold)
        elif self._type in ['EMLP']:
            return self.predict_feature_mlp(probe_features, drop_frame=1, threshold=threshold)
        else: return (None, None, None)

    def save_features(self, file_name='saves/train_feats.py'):
        np.save(file_name, [self._features, self._labels])

    def save_model(self, file_name='saves/model.npy'):
        np.save(file_name, [self._labels, self._models, self._type])

    def trainPLS(self, components=10, iterations=500):
        from sklearn.cross_decomposition import PLSRegression
        self._models = list()
        self._type = 'OAAPLS'
        print('Training One-Against-All PLS classifiers')
        for label in self.get_classes():
            classifier = PLSRegression(n_components=components, max_iter=iterations)
            boolean_label = [+1.0 if label == lab else -1.0 for lab in self._labels]
            model = classifier.fit(np.array(self._features), np.array(boolean_label))
            self._models.append((model, label))
        # self.save_model(file_name='saves/pls_model.npy')

    def trainSVM(self, cpar=1.0, mode='libsvm', kernel_type='linear', iterations=5000, verbose=False):
        from sklearn.svm import LinearSVR, SVR, NuSVR
        self._type = 'OAASVM'
        self._models = list()
        print('Training One-Against-All SVM classifiers')
        for label in self.get_classes():
            if mode == 'libsvm': classifier = SVR(C=cpar, kernel=kernel_type, verbose=verbose)
            elif mode == 'liblinear': classifier = LinearSVR(C=cpar, max_iter=iterations, verbose=verbose)
            elif mode =='libsvm-nu': classifier = NuSVR(nu=0.5, C=cpar, kernel=kernel_type, verbose=verbose)            
            boolean_label = [label == lab for lab in self._labels]
            model = classifier.fit(np.array(self._features), np.array(boolean_label))
            self._models.append((model, label))
        # self.save_model(file_name='saves/svm_model.npy')

    def trainOCSVM(self, samples4model=15000, pos_label='live', neg_label='spoof', nu_val=0.01, kernel_type='linear', gamma_val=0.1):
        assert(len(self.get_classes()) == 1)
        from sklearn.svm import OneClassSVM
        self._models = list()
        self._neg_label = neg_label
        self._pos_label = pos_label
        self._type = 'OCSVM'
        print('Training One-Class SVM')
        rand_features, _ = self.__feature_sampling(num_samples=samples4model)
        classifier = OneClassSVM(gamma=gamma_val, kernel=kernel_type, nu=nu_val, shrinking=False, verbose=True)
        model = classifier.fit(np.array(rand_features))
        self._models.append(model)

    def trainEPLS(self, models=50, samples4model=500, pos_label='live', neg_label='spoof', components=10, iterations=500):
        from sklearn.cross_decomposition import PLSRegression
        self._models = list()
        self._neg_label = neg_label
        self._pos_label = pos_label
        self._type = 'EPLS'
        print('Training an Ensemble of PLS classifiers')
        for index in range(models):
            classifier = PLSRegression(n_components=components, max_iter=iterations)
            rand_features, rand_labels = self.__feature_sampling(num_samples=samples4model)
            boolean_label = [+1.0 if self._pos_label == lab else -1.0 for lab in rand_labels]
            model = classifier.fit(np.array(rand_features), np.array(boolean_label))
            self._models.append(model)
            print(' -> Training model %3d with %d random samples' % (index + 1, samples4model))
        print('Feature Shape', rand_features[0].shape)
        # self.save_model(file_name='saves/epls_model.npy')

    def trainESVM(self, models=50, samples4model=500, pos_label='live', neg_label='spoof', cpar=1.0, mode='libsvm', kernel_type='linear', iterations=500, verbose=False):
        from sklearn.svm import LinearSVR, SVR, NuSVR
        self._models = list()
        self._neg_label = neg_label
        self._pos_label = pos_label
        self._type = 'ESVM'
        print('Training an Ensemble of SVM classifiers')
        for index in range(models):
            if mode == 'libsvm': classifier = SVR(C=cpar, kernel=kernel_type, verbose=verbose)
            elif mode == 'liblinear': classifier = LinearSVR(C=cpar, max_iter=iterations, verbose=verbose)
            elif mode =='libsvm-nu': classifier = NuSVR(nu=0.5, C=cpar, kernel=kernel_type, verbose=verbose) 
            rand_features, rand_labels = self.__feature_sampling(num_samples=samples4model)
            boolean_label = [+1.0 if self._pos_label == lab else -1.0 for lab in rand_labels]
            model = classifier.fit(np.array(rand_features), np.array(boolean_label))
            self._models.append(model)
            print(' -> Training model %3d with %d random samples' % (index + 1, samples4model))
        print('Feature Shape', rand_features[0].shape)
        # self.save_model(file_name='saves/esvm_model.npy')
        
    def trainEMLP(self, models=50, samples4model=500, pos_label='live', neg_label='spoof'):
        def getModel(input_shape, nclasses=2):
            model = keras_sequential()
            model.add(keras_batch_norm(input_shape=input_shape))
            model.add(keras_dense(units=128, activation='relu'))
            model.add(keras_dropout(rate=0.5))
            # model.add(keras_batch_norm())
            # model.add(keras_dense(units=128, activation='relu'))
            # model.add(keras_dropout(rate=0.5))
            # model.add(keras_batch_norm())
            model.add(keras_dense(units=64, activation='relu'))
            model.add(keras_dropout(rate=0.5))
            model.add(keras_dense(units=nclasses, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        self._models = list()
        self._neg_label = neg_label
        self._pos_label = pos_label
        self._type = 'EMLP'
        print('Training an Ensemble of MLP classifiers')
        for index in range(models):
            rand_features, rand_labels = self.__feature_sampling(num_samples=samples4model)
            bool_labels = [+1.0 if self._pos_label == lab else 0.0 for lab in rand_labels]
            cate_labels = keras_np_utils.to_categorical(bool_labels, 2)
            model = getModel(input_shape=rand_features[0].shape)
            model.fit(np.asarray(rand_features), np.asarray(cate_labels), batch_size=10, epochs=100, verbose=0, validation_split=0.1)
            self._models.append(model)
            print(' -> Training model %3d with %d random samples' % (index + 1, samples4model))
        print('Feature Shape', rand_features[0].shape)
        # self.save_model(file_name='saves/emlp_model.npy')


"""# Execution"""

# Store all-interation results
max_neg_values = list()
min_pos_values = list()
result_errors = dict()
result_labels = list()
result_scores = list()
thresholds = dict()

# Split dataset into train and test sets
dataset_dict = tuple_to_dict(DATASET_FILE)
dataset_inds = {z_data.split('/')[-2] for (y_data, z_data) in dataset_dict.keys()}


# SWAX (hog, lbp-1, lbp-2, glcm, total):  32 795 795 48 1670

if DESCRIPTOR != 'combined':
    if DESCRIPTOR == 'hog':
        beg_index = 0
        end_index = 32
    elif DESCRIPTOR == 'lbp1':
        beg_index = 32
        end_index = 827
    elif DESCRIPTOR == 'lbp2':
        beg_index = 827
        end_index = 1622
    elif DESCRIPTOR == 'lbp':
        beg_index = 32
        end_index = 1622
    elif DESCRIPTOR == 'glcm':
        beg_index = 1622
        end_index = 1670

    dataset_dict = trim_descriptor_data(dataset_dict, beg_index, end_index)

for (index, individual) in enumerate(dataset_inds):
    print('> ITERATION {}: {}'.format(index + 1, individual.upper()))
    max_neg_value = -10.0
    min_pos_value = +10.0

    if SCENARIO == 'one':
        c_train_dict, c_probe_dict = swax_protocol_01(dataset_dict, individual, max_frames=60)
        c_train_dict, c_devel_dict = set_aside_validation(c_train_dict, percent=ASIDE)
    elif SCENARIO == 'two':
        c_train_dict, c_probe_dict = swax_protocol_02(dataset_dict, individual, medium_out=index+1, max_frames=MAX_FRAMES, skip_frames=DROP_FRAMES)
        c_train_dict, c_devel_dict = set_aside_validation(c_train_dict, percent=ASIDE)
    elif SCENARIO == 'three':
        c_train_dict, c_probe_dict = swax_protocol_03(dataset_dict, individual, category_out=index+2, max_frames=MAX_FRAMES, skip_frames=DROP_FRAMES)
        c_train_dict, c_devel_dict = set_aside_validation(c_train_dict, percent=ASIDE)
    elif SCENARIO == 'four':
        c_train_dict, c_probe_dict = swax_protocol_03(dataset_dict, individual, category_out=index+2, max_frames=MAX_FRAMES, skip_frames=DROP_FRAMES)
        c_train_dict, c_devel_dict = set_aside_validation(c_train_dict, percent=ASIDE)


    # Change into a binary problem
    c_train_dict, c_probe_dict = binarize_label(c_train_dict, c_probe_dict, input_label='real', pos_label='live', neg_label='spoof')

    # Print data size
    print('Probe size:', len(c_probe_dict), sum([1 for (y_data, z_data) in c_probe_dict.keys() if y_data == 'live']), sum([1 for (y_data, z_data) in c_probe_dict.keys() if y_data != 'live']))
    print('Train size:', len(c_train_dict), sum([1 for (y_data, z_data) in c_train_dict.keys() if y_data == 'live']), sum([1 for (y_data, z_data) in c_train_dict.keys() if y_data != 'live']))
    print('Valid size:', len(c_devel_dict), sum([1 for (y_data, z_data) in c_devel_dict.keys() if y_data == 'live']), sum([1 for (y_data, z_data) in c_devel_dict.keys() if y_data != 'live']))

    # Instantiate SpoofDet class
    spoofDet = FaceSpoofing()
    spoofDet.import_features(feature_dict=c_train_dict)
    
    # SWAX (hog, lbp-1, lbp-2, glcm, total): 32 795 795 48 1670

    # Check whether class is ready to continue
    print('Classes: ', spoofDet.get_classes())
    assert('live' in spoofDet.get_classes())
    
    if BAGGING:
        # spoofDet.trainEMLP(models=BAGGING, samples4model=INSTANCES)
        spoofDet.trainEPLS(models=BAGGING, samples4model=INSTANCES)
        # spoofDet.trainESVM(models=BAGGING, samples4model=INSTANCES) 
    else:
        # spoofDet.trainPLS(components=10, iterations=1000)
        spoofDet.trainOCSVM(nu_val=0.05, kernel_type='rbf', gamma_val=0.1)

    # Define APCER/BPCER variables
    instances = list({y_data for (y_data, z_data) in c_probe_dict.keys()})
    counter_dict = {y_data:0.0 for (y_data, z_data) in c_probe_dict.keys()}
    mistake_dict = {y_data:0.0 for (y_data, z_data) in c_probe_dict.keys()}

    # Define lists to plot charts
    result = dict()
    result['labels'] = list()
    result['scores'] = list()

    c_probe_tuples = list(c_probe_dict.keys())
    c_train_tuples = list(c_train_dict.keys())

    # THRESHOLD: Predict samples
    validation_labels = list()
    validation_scores = list()
    video_counter = 0
    for (label, path) in c_probe_tuples[:1000]:
        probe_feats = c_probe_dict[(label, path)]
        pred_label, pred_score = spoofDet.predict_feature(probe_feats, threshold=0.55)
        validation_labels.append(+1) if label == 'live' else validation_labels.append(-1)
        validation_scores.append(pred_score)
        video_counter += 1
        print(video_counter, '>>', path, label, '>>', {pred_label:pred_score})
    precision, recall, threshold = precision_recall_curve(validation_labels, validation_scores)
    fmeasure = [(thr, (2.0 * (pre * rec) / (pre + rec))) for pre, rec, thr in zip(precision[:-1], recall[:-1], threshold)]
    fmeasure.sort(key=lambda tup:tup[1], reverse=True)
    best_threshold = fmeasure[0][0]
    print('SELECTED THRESHOLD', best_threshold)

    # TEST: Predict samples
    video_counter = 0
    for (label, path) in c_probe_tuples:
        counter_dict[label] += 1
        probe_feats = c_probe_dict[(label, path)]
        pred_label, pred_score = spoofDet.predict_feature(probe_feats, threshold=best_threshold)
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

    # Compute average APCER and BPCER
    print('\n')
    print(SCENARIO.upper())
    print('BAGGING:', BAGGING, 'INSTANCES:', INSTANCES)
    print('MAX_NEG_VALUES:', max_neg_values, 'MIN_POS_VALUES:', min_pos_values)
    print("ERROR RESULT per ITERATION:", result_errors)
    for label in result_errors.keys():
        error_avg = np.mean(result_errors[label])
        error_std = np.std(result_errors[label])
        print("ERROR RESULT (label, avg, std):", label, error_avg, error_std)
    print('------------------------------------------------------------------\n')