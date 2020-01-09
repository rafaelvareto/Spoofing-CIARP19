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

# from keras.models import Sequential 
# from keras.utils import np_utils
# from deep_learning import DeepLearning
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
        sample_gray = cv.cvtColor(sample_image, cv.COLOR_BGR2GRAY)
        sample_hsvs = cv.cvtColor(sample_image, cv.COLOR_BGR2HSV)
        sample_ycrc = cv.cvtColor(sample_image, cv.COLOR_BGR2YCrCb)

        sample_noise = self.get_residual_noise(gray_img=sample_gray, filter_type='median')
        sample_spectrum = self.get_fourier_spectrum(noise_img=sample_noise)
        
        sample_featureA = self._descriptor.get_hog_feature(image=sample_gray, pixel4cell=(96,96), cell4block=(1,1), orientation=8)
        sample_featureB = self._descriptor.get_lbp_ch_feature(image=sample_hsvs, bins=265, points=8, radius=1)
        sample_featureC = self._descriptor.get_lbp_ch_feature(image=sample_ycrc, bins=265, points=8, radius=1)
        sample_featureD = self._descriptor.get_glcm_feature(image=sample_spectrum, dists=[1,2], shades=20)
        sample_feature = np.concatenate((sample_featureA, sample_featureB, sample_featureC, sample_featureD), axis=0)
        if show:
            cv.imshow('spectrum', cv.normalize(sample_spectrum, 0, 255, cv.NORM_MINMAX))
            cv.waitKey(1)
        return sample_feature

    def import_features(self, feature_dict):
        for ((label, path), features) in feature_dict.items():
            print('Imported Features: ', (label, path), len(features), len(features[0]))
            for feat in features:
                self._features.append(np.asarray(feat))
                self._labels.append(label)
                self._paths.append(path)

    def load_model(self, file_name='saves/model.npy'):
        self._labels, self._models, self._type = np.load(file_name)

    def obtain_image_features(self, folder_path, dataset_tuple, new_size=None, file_name='saves/image_features.npy'):
        if new_size is not None:
            self._size = new_size
        for (path, label) in dataset_tuple:
            sample_path = os.path.join(folder_path, path)
            sample_image = cv.imread(sample_path, cv.IMREAD_COLOR)
            scaled_image = cv.resize(sample_image, (self._size[0], self._size[1]), interpolation=cv.INTER_AREA)
            feature = self.gray2feat_pipeline(scaled_image)
            if not np.any(np.isnan(feature)):
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
                        if not np.any(np.isnan(feature)):
                            self._features.append(feature)
                            self._labels.append(label)
                else:
                    break
                frame_counter += 1
            video_counter += 1
            np.save(file_name, [self._features, self._labels])

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

    def predict_feature_bag(self, probe_features, drop_frame, threshold):
        mean_list = list()
        for (index, feature) in enumerate(probe_features):
            if index % drop_frame == 0:
                results = [float(model.predict(np.array([feature]))) for model in self._models]
                binnary = [+1.0 if result > 0.0 else 0.0 for result in results]
                mean_list.append(sum(binnary) / len(binnary))
        score = np.mean(mean_list)
        label = self._pos_label if score >= threshold else self._neg_label
        return (label, score)

    def predict_feature_mlp(self, probe_features, drop_frame, threshold):
        mean_list = list()
        for (index, feature) in enumerate(probe_features):
            print(feature.shape, feature)
            if index % drop_frame == 0:
                afeature = np.asarray(feature)
                # print(afeature, afeature.shape)
                # results = [float(model.predict(afeature.reshape(1, afeature.shape[0]))) for model in self._models]
                results = [model.predict(afeature[0:1]) for model in self._models]
                binnary = [+1.0 if result > 0.0 else 0.0 for result in results]
                mean_list.append(sum(binnary) / len(binnary))
        score = np.mean(mean_list)
        label = self._pos_label if score >= threshold else self._neg_label
        return (label, score)

    def predict_feature(self, probe_features, drop_frame=10, threshold=0.50):
        if self._type in ['OAAPLS', 'OAASVM']:
            return self.predict_feature_oaa(probe_features, drop_frame, threshold)
        elif self._type in ['EPLS', 'ESVM']:
            return self.predict_feature_bag(probe_features, drop_frame, threshold)
        elif self._type in ['EMLP']:
            return self.predict_feature_mlp(probe_features, drop_frame, threshold)
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
        self.save_model(file_name='saves/pls_model.npy')

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
        self.save_model(file_name='saves/svm_model.npy')

    def trainEPLS(self, models=50, samples4model=50, pos_label='live', neg_label='spoof', components=10, iterations=500):
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
        self.save_model(file_name='saves/epls_model.npy')

    def trainESVM(self, models=50, samples4model=50, pos_label='live', neg_label='spoof', cpar=1.0, mode='libsvm', kernel_type='linear', iterations=5000, verbose=False):
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
        self.save_model(file_name='saves/esvm_model.npy')
        
    def trainEMLP(self, models=50, samples4model=50, pos_label='live', neg_label='spoof'):
        import keras
        import keras.models
        import keras.backend.tensorflow_backend as keras_backend
        import tensorflow
        from keras.models import Sequential as keras_sequential
        from keras.layers import Dense as keras_dense 
        from keras.layers import Dropout as keras_dropout
        from keras.utils import np_utils as keras_np_utils
        def getModel(input_shape, nneuros=64, nclasses=2):
            model = keras_sequential()
            model.add(keras_dense(nneuros, activation='relu', input_shape=input_shape))
            model.add(keras_dropout(0.2))
            model.add(keras_dense(nclasses, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])#RMSprop()
            return model
        self._models = list()
        self._neg_label = neg_label
        self._pos_label = pos_label
        self._type = 'EMLP'
        print('Training an Ensemble of MLP classifiers')
        for index in range(models):
            rand_features, rand_labels = self.__feature_sampling(num_samples=samples4model)
            bool_labels = [+1.0 if self._pos_label == lab else -1.0 for lab in rand_labels]
            cate_labels = keras_np_utils.to_categorical(bool_labels, 2)
            model = getModel(input_shape=rand_features[0].shape)
            model.fit(np.array(rand_features), np.array(cate_labels), batch_size=40, nb_epoch=100, verbose=0)
            self._models.append(model)
            print(' -> Training model %3d with %d random samples' % (index + 1, samples4model))
        print('Feature Shape', rand_features[0].shape)
        self.save_model(file_name='saves/esvm_model.npy')
