import cv2 as cv
import numpy as np
import tensorflow as tf

from keras.layers import Conv2D, Input,MaxPool2D, Reshape, Activation, Flatten, Dense, Permute
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU

import tools_matrix as tools

class FaceDetection:
    def __init__(self, threshold=[0.5,0.4,0.6], pnet_name=r'weights/12net.h5', rnet_name=r'weights/24net.h5', onet_name=r'weights/48net.h5'):
        print('Face Detection Class')
        self._threshold = threshold
        self._pnet = self.__create_Kao_Pnet(pnet_name)
        self._rnet = self.__create_Kao_Rnet(rnet_name)
        self._onet = self.__create_Kao_Onet(onet_name)

    def detectFace(self, image):
        copy_img = (image.copy() - 127.5) / 127.5
        origin_h, origin_w, ch = copy_img.shape
        scales = tools.calculateScales(image)
        out = []

        # Execution of FIRST network
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv.resize(copy_img, (ws, hs))
            input = scale_img.reshape(1, *scale_img.shape)
            ouput = self._pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
            out.append(ouput)
        image_num = len(scales)
        rectangles = []
        for idx in range(image_num):
            cls_prob = out[idx][0][0][:, :, 1]  # idx = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
            roi = out[idx][1][0]
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            cls_prob = np.swapaxes(cls_prob, 0, 1)
            roi = np.swapaxes(roi, 0, 2)
            rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[idx], origin_w, origin_h, self._threshold[0])
            rectangles.extend(rectangle)
        rectangles = tools.NMS(rectangles, 0.7, 'iou')

        if len(rectangles) == 0:
            return rectangles

        # Execution of SECOND network
        crop_number = 0
        out = []
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)
            crop_number += 1
        predict_24_batch = np.array(predict_24_batch)
        out = self._rnet.predict(predict_24_batch)
        cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
        cls_prob = np.array(cls_prob)  # convert to numpy
        roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
        roi_prob = np.array(roi_prob)
        rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, self._threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # Execution of THIRD network
        crop_number = 0
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)
            crop_number += 1
        predict_batch = np.array(predict_batch)
        output = self._onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]  # index
        rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, self._threshold[2])

        return [rect[0:5] for rect in rectangles]

    def __create_Kao_Onet(self, weight_path):
        input = Input(shape = [48,48,3])
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1,2],name='prelu1')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        
        x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1,2],name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        
        x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1,2],name='prelu3')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
        
        x = PReLU(shared_axes=[1,2],name='prelu4')(x)
        x = Permute((3,2,1))(x)
        x = Flatten()(x)
        x = Dense(256, name='conv5') (x)
        x = PReLU(name='prelu5')(x)

        classifier = Dense(2, activation='softmax',name='conv6-1')(x)
        bbox_regress = Dense(4,name='conv6-2')(x)
        landmark_regress = Dense(10,name='conv6-3')(x)
        model = Model([input], [classifier, bbox_regress, landmark_regress])
        model.load_weights(weight_path, by_name=True)
        return model

    def __create_Kao_Rnet(self, weight_path):
        input = Input(shape=[24, 24, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
        x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
        x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

        x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
        
        x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
        x = Permute((3, 2, 1))(x)
        x = Flatten()(x)
        x = Dense(128, name='conv4')(x)
        x = PReLU( name='prelu4')(x)
        
        classifier = Dense(2, activation='softmax', name='conv5-1')(x)
        bbox_regress = Dense(4, name='conv5-2')(x)
        model = Model([input], [classifier, bbox_regress])
        model.load_weights(weight_path, by_name=True)
        return model

    def __create_Kao_Pnet(self, weight_path):
        input = Input(shape=[None, None, 3])
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
        x = MaxPool2D(pool_size=2)(x)
        
        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1,2],name='PReLU2')(x)
        
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1,2],name='PReLU3')(x)
        
        classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
        bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)
        model = Model([input], [classifier, bbox_regress])
        model.load_weights(weight_path, by_name=True)
        return model


class FaceDetectionCV:
    def __init__(self):
        self.face_cascade = cv.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')
    
    def detectFace(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        rectangles = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return rectangles