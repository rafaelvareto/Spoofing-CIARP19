'''
https://github.com/allansp84/visualrhythm-antispoofing/tree/master/src
https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
'''

import copy
import cv2 as cv
import numpy as np
import os
import random

from keras.models import Sequential 
from keras.utils import np_utils
from deep_learning import DeepLearning
from descriptors import Descriptors


class FaceSpoofing:
    def __init__(self):
        print('Face Spoofing Class')
        self._color_space = 0
        self._descriptor = Descriptors()
        self._dictionary = dict()
        self._gaussian_var = 2.0
        self._features = list()
        self._fourier = list()
        self._images = list()
        self._kernel_size = 7
        self._labels = list()
        self._models = None
        self._size = (640, 360)
        self._type = 'None'
        self._vr_height = 1
        self._vr_width = 30

    def __build_dictionary(self):
        lab_dict = {label:number for (number,label) in zip(range(self.get_num_classes()), self.get_classes())}
        num_dict = {number:label for (number,label) in zip(range(self.get_num_classes()), self.get_classes())}
        self._dictionary = dict( list(lab_dict.items()) + list(num_dict.items()) )
        print(self._dictionary)

    def __manage_results(self, dictionary, score_list):
        for (label, result) in score_list:
            if label in dictionary:
                dictionary[label].append(result)
            else:
                dictionary[label] = [result, ]
        return dictionary

    def __mean_and_return(self, dictionary):
        new_list = {key:float(np.mean(value)) for (key, value) in dictionary.items()}
        return new_list

    def __mean_and_sort(self, dictionary):
        new_list = [(key,float(np.mean(value))) for (key, value) in dictionary.items()]
        new_list.sort(key=lambda tup:tup[1], reverse=True)
        return new_list

    def __channel_swap(self, image): 
        spare = copy.copy(image)
        image[:, :, 0] = spare[:, :, 2]
        image[:, :, 2] = spare[:, :, 0]
        return image

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

    def get_fourier_spectrum(self, noise_img):
        fft_img = np.fft.fft2(noise_img)
        fts_img = np.fft.fftshift(fft_img)
        mag_img = np.abs(fts_img)
        log_img = np.log(1 + mag_img)
        return log_img

    def get_num_classes(self):
        categories = set(self._labels)
        return len(categories)

    def gray2spec_pipeline(self, sample_image, show=False): 
        sample_gray = self.get_gray_image(color_img=sample_image) 
        sample_noise = self.get_residual_noise(gray_img=sample_gray, filter_type='median') 
        sample_spectrum = self.get_fourier_spectrum(noise_img=sample_noise) 
        if show: 
            cv.imshow('spectrum', cv.normalize(sample_spectrum, 0, 255, cv.NORM_MINMAX)) 
            cv.waitKey(1) 
        return sample_spectrum 

    def gray2feat_pipeline(self, sample_image, show=False):
        sample_gray = self.get_gray_image(color_img=sample_image)
        sample_noise = self.get_residual_noise(gray_img=sample_gray, filter_type='median')
        sample_spectrum = self.get_fourier_spectrum(noise_img=sample_noise)
        sample_featureA = self._descriptor.get_hog_feature(image=sample_gray, pixel4cell=(64,64), cell4block=(1,1), orientation=8)
        sample_featureB = self._descriptor.get_glcm_feature(image=sample_spectrum, dists=[1,2], shades=20)
        sample_feature = np.concatenate((sample_featureA, sample_featureB), axis=0)
        if show:
            cv.imshow('spectrum', cv.normalize(sample_spectrum, 0, 255, cv.NORM_MINMAX))
            cv.waitKey(1)
        return sample_feature

    def obtain_image_features(self, folder_path, dataset_tuple, new_size=None, file_name='saves/image_features.npy'):
        if new_size is not None:
            self._size = new_size
        for (path, label) in dataset_tuple:
            sample_path = os.path.join(folder_path, path)
            sample_image = cv.imread(sample_path, cv.IMREAD_COLOR)
            scaled_image = cv.resize(sample_image, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
            feature = self.gray2feat_pipeline(scaled_image)
            self._features.append(feature)
            self._labels.append(label)
        np.save(file_name, [self._features, self._labels])

    def obtain_video_features(self, folder_path, dataset_tuple, frame_drop=1, max_frames=60, new_size=None, file_name='saves/video_features.npy', verbose=False):
        video_counter = 0
        if new_size is not None:
            self._size = new_size
        for (path, label) in dataset_tuple:
            if verbose:
                print(video_counter + 1, path, label)
            frame_counter = 0
            sample_path = os.path.join(folder_path, path)
            sample_video = cv.VideoCapture(sample_path)
            while(sample_video.isOpened()):
                ret, sample_frame = sample_video.read()
                if ret and frame_counter <= max_frames:
                    if frame_counter % frame_drop == 0:
                        scaled_frame = cv.resize(sample_frame, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
                        feature = self.gray2feat_pipeline(scaled_frame)
                        self._features.append(feature)
                        self._labels.append(label)
                else:
                    break
                frame_counter += 1
            video_counter += 1
            np.save(file_name, [self._features, self._labels])

    def load_model(self, file_name='saves/model.npy'):
        self._labels, self._models, self._type = np.load(file_name)

    def predict_image(self, probe_image):
        if self._type == 'PLS' or self._type == 'SVM':
            class_dict = dict()
            scaled_image = cv.resize(probe_image, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
            feature = self.gray2feat_pipeline(scaled_image)
            results = [float(model[0].predict(np.array([feature]))) for model in self._models]
            labels = [model[1] for model in self._models]
            scores = list(map(lambda left,right:(left,right), labels, results))
            class_dict = self.__manage_results(class_dict, scores)
            return self.__mean_and_sort(class_dict)
        elif self._type == 'CNN': 
            pass 
        else:
            raise ValueError('Error predicting probe image') 

    def predict_video(self, probe_video, frame_drop=10):
        if self._type == 'PLS' or self._type == 'SVM':
            frame_counter = 0
            class_dict = dict()
            while(probe_video.isOpened()):
                ret, probe_frame = probe_video.read()
                if ret:
                    if frame_counter % frame_drop == 0:
                        scaled_frame = cv.resize(probe_frame, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
                        feature = self.gray2feat_pipeline(scaled_frame)
                        results = [float(model[0].predict(np.array([feature]))) for model in self._models]
                        labels = [model[1] for model in self._models]
                        scores = list(map(lambda left,right:(left,right), labels, results))
                        class_dict = self.__manage_results(class_dict, scores)
                    elif self._type == 'CNN': 
                        scaled_image = cv.resize(probe_frame, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
                        gray_image = self.get_gray_image(scaled_image) 
                        spec_image = self.gray2spec_pipeline(scaled_image)
                        if self._size[2] == 1:
                            array_image = np.asarray([gray_image])
                        elif self._size[2] == 3:
                            array_image = np.asarray([scaled_image])
                        results = self._models.predict(array_image).ravel()
                        labels = [self._dictionary[index] for index in range(len(results))]
                        scores = list(map(lambda left,right:(left,right), labels, results))
                        class_dict = self.__manage_results(class_dict, scores)
                    else:
                        raise ValueError('Error predicting probe video')
            else:
                break
            frame_counter += 1 
        return self.__mean_and_sort(class_dict)

    def save_model(self, file_name='saves/model.npy'):
        np.save(file_name, [self._labels, self._models, self._type])

    def trainPLS(self, components=10, iterations=500):
        self._type = 'PLS'
        self._models = list()
        from sklearn.cross_decomposition import PLSRegression
        for label in self.get_classes():
            classifier = PLSRegression(n_components=components, max_iter=iterations)
            boolean_label = [label == lab for lab in self._labels]
            model = classifier.fit(np.array(self._features), np.array(boolean_label))
            self._models.append((model, label))
        self.save_model(file_name='pls_model.npy') 

    def trainSVM(self, kernel_type='rbf', verbose=False):
        self._type = 'SVM'
        self._models = list()
        from sklearn.svm import SVR
        for label in self.get_classes():
            classifier = SVR(C=1.0, kernel=kernel_type, verbose=verbose)
            boolean_label = [label == lab for lab in self._labels]
            model = classifier.fit(np.array(self._features), np.array(boolean_label))
            self._models.append((model, label))
        self.save_model(file_name='svm_model.npy') 

    def trainCNN(self, batch=128, epoch=20, weightsPath=None): 
        self._type = 'CNN'
        self.__build_dictionary()
        int_labels = [self._dictionary[label] for label in self._labels]
        cat_labels = np_utils.to_categorical(int_labels, self.get_num_classes())
        self._models = DeepLearning.build_LeNet(width=self._size[0], height=self._size[1], depth=self._size[2], nclasses=self.get_num_classes(), weightsPath=weightsPath) 
        if weightsPath is None:
            self._models.fit(np.array(self._images), np.array(cat_labels), batch_size=batch, epochs=epoch, verbose=1)
            self._models.save_weights('cnn_model.h5', overwrite=True) 
        
        