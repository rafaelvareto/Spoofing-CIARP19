'''
https://github.com/allansp84/visualrhythm-antispoofing/tree/master/src
https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
'''

import cv2 as cv
import numpy as np
import os
import random

# from keras.utils import np_utils
# from deep_learning import DeepLearning
from descriptors import Descriptors


class FaceSpoofing:
    def __init__(self):
        print('Face Spoofing Class')
        self._color_space = 0
        self._descriptor = Descriptors()
        self._gaussian_var = 2.0
        self._features = list()
        self._images = list()
        self._kernel_size = 7
        self._labels = list()
        self._models = None
        self._type = 'None'
        self._vr_height = 1
        self._vr_width = 30

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

    def obtain_image_features(self, folder_path, dataset_tuple):
        for (path, label) in dataset_tuple:
            sample_path = os.path.join(folder_path, path)
            sample_image = cv.imread(sample_path, cv.IMREAD_COLOR)
            feature = self.gray2feat_pipeline(sample_image)
            self._features.append(feature)
            self._labels.append(label)

    def obtain_video_features(self, folder_path, dataset_tuple, frame_drop=10, verbose=False):
        for (path, label) in dataset_tuple:
            if verbose:
                print(path, label)
            frame_counter = 0
            sample_path = os.path.join(folder_path, path)
            sample_video = cv.VideoCapture(sample_path)
            while(sample_video.isOpened()):
                ret, sample_frame = sample_video.read()
                if ret:
                    if frame_counter % frame_drop == 0:
                        feature = self.gray2feat_pipeline(sample_frame)
                        self._features.append(feature)
                        self._labels.append(label)
                else:
                    break
                frame_counter += 1

    def load_model(self, file_name='model.npy'):
        self._labels, self._models, self._type = np.load(file_name)

    def predict_image(self, probe_image):
        if self._type == 'PLS' or self._type == 'SVM':
            class_dict = dict()
            feature = self.gray2feat_pipeline(probe_image)
            results = [float(model[0].predict(np.array([feature]))) for model in self._models]
            labels = [model[1] for model in self._models]
            scores = list(map(lambda left,right:(left,right), labels, results))
            class_dict = self.__manage_results(class_dict, scores)
            return self.__mean_and_sort(class_dict)
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
                        feature = self.gray2feat_pipeline(probe_frame)
                        results = [float(model[0].predict(np.array([feature]))) for model in self._models]
                        labels = [model[1] for model in self._models]
                        scores = list(map(lambda left,right:(left,right), labels, results))
                        class_dict = self.__manage_results(class_dict, scores)
                else:
                    break
                frame_counter += 1 
            return self.__mean_and_sort(class_dict)
        else:
            raise ValueError('Error predicting probe video')

    def save_model(self, file_name='model.npy'):
        np.save(file_name, [self._labels, self._models, self._type])

    def trainDL(self, nclasses=3):
        self._type = 'DL'
        self._model = DeepLearning.build_LeNet(width=96, height=96, depth=1, classes=nclasses)
        train_labels = np_utils.to_categorical(self._labels, nclasses)
        pass

    def trainPLS(self, components=10, iterations=500):
        self._models = list()
        self._type = 'PLS'
        from sklearn.cross_decomposition import PLSRegression
        for label in self.get_classes():
            classifier = PLSRegression(n_components=components, max_iter=iterations)
            boolean_label = [label == lab for lab in self._labels]
            model = classifier.fit(np.array(self._features), np.array(boolean_label))
            self._models.append((model, label))

    def trainSVM(self, kernel_type='rbf', verbose=False):
        self._models = list()
        self._type = 'SVM'
        from sklearn.svm import SVR
        for label in self.get_classes():
            classifier = SVR(C=1.0, kernel=kernel_type, verbose=verbose)
            boolean_label = [label == lab for lab in self._labels]
            model = classifier.fit(np.array(self._features), np.array(boolean_label))
            self._models.append((model, label))

