import argparse
import cv2 as cv
import json
import numpy as np
import os
import os.path
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("..")
from descriptors import Descriptors

HOME = os.path.expanduser("~")

def load_txt_file(file_name):
    this_file = open(file_name, 'r')
    this_list = list()
    for line in this_file:
        line = line.rstrip()
        components = line.split()
        this_list.append(components)
    return this_list

def load_face_file(file_name):
    # (x,y) locations of the upper left corner and the botton right corner
    this_file = open(file_name, 'r', encoding='utf-8', errors='ignore')
    this_list = list()
    for line in this_file:
        print(line)
        line = line.rstrip()
        components = line.split()
        this_list.append(components)
    return this_list 

def get_cropped_face(color_img, eye_tuple, scale=0.50, h_margin=0, v_margin=0):
    # frame number, (x,y) locations of the upper left corner and width and height values
    if (len(eye_tuple) < 5) or (eye_tuple[1] == eye_tuple[2] == eye_tuple[3] == eye_tuple[4]):
        return color_img
    else:
        bbox = [int(float(item) * scale) for item in eye_tuple]
        bbox = [0 if item < 0 else item for item in bbox]
        face_crop = color_img[bbox[2]-v_margin:bbox[2]+v_margin+bbox[4], bbox[1]-h_margin:bbox[1]+h_margin+bbox[3]]
        return face_crop

def get_fourier_spectrum(noise_img):
    fft_img = np.fft.fft2(noise_img)
    fts_img = np.fft.fftshift(fft_img)
    mag_img = np.abs(fts_img)
    log_img = np.log(1 + mag_img)
    return log_img

def get_residual_noise(gray_img, filter_type='median', kernel_size=7, gaussian_var=2.0):
    if filter_type == 'median':
        blurred = cv.medianBlur(src=gray_img, ksize=kernel_size)
        noise = cv.subtract(gray_img, blurred)
    elif filter_type == 'gaussian':
        blurred = cv.GaussianBlur(src=gray_img, ksize=(kernel_size,kernel_size), sigmaX=gaussian_var, sigmaY=0.0)
        noise = cv.subtract(gray_img, blurred)
    else:
        raise ValueError('ERROR: Two smoothing methods available: median and gaussian')
    return noise

def obtain_video_features(folder_path, dataset_tuple, frame_drop=1, scale=0.5, file_name='video_features.npy', saveCopy=False, show=False, verbose=False):
    descriptor = Descriptors()
    feature_list = list()
    label_list = list()
    path_list = list()

    if os.path.isfile(file_name):
        print('>> Importing', file_name)
        feature_list, label_list, path_list = np.load(file_name)
        feature_list = list(feature_list)
        label_list = list(label_list)
        path_list = list(path_list)

    inner_counter = overall_counter = 0
    for (path, label) in dataset_tuple:
        if path not in path_list:
            frame_counter = 0
            probe_fourcc = cv.VideoWriter_fourcc(*'MP42') 

            read_path = os.path.join(folder_path, path)
            read_video = cv.VideoCapture(read_path)

            annt_path = os.path.join(folder_path, path.replace('.avi', '.face'))
            annt_tuples = load_face_file(annt_path)

            if verbose:
                print(overall_counter + 1, inner_counter + 1, path, label)
            if saveCopy:
                spec_video = cv.VideoWriter(read_path.replace('.mov', '_spec.avi'), probe_fourcc, 20.0, size, isColor=False)
                tiny_video = cv.VideoWriter(read_path.replace('.mov', '_tiny.avi'), probe_fourcc, 20.0, size, isColor=True)

            while(read_video.isOpened()):
                ret, read_frame = read_video.read()
                if ret:
                    if frame_counter % frame_drop == 0:
                        size = [int(scale * read_frame.shape[1]), int(scale * read_frame.shape[0])]
                        read_color = cv.resize(read_frame, (size[0], size[1]), interpolation=cv.INTER_AREA)

                        read_greyd = cv.cvtColor(read_color, cv.COLOR_BGR2GRAY)
                        read_hsvch = get_cropped_face(cv.cvtColor(read_color, cv.COLOR_BGR2HSV), scale=scale, eye_tuple=annt_tuples[frame_counter])
                        read_ycrcb = get_cropped_face(cv.cvtColor(read_color, cv.COLOR_BGR2YCrCb), scale=scale, eye_tuple=annt_tuples[frame_counter])
                        cv.imshow('face', get_cropped_face(read_color, scale=scale, eye_tuple=annt_tuples[frame_counter]))
                        cv.waitKey(0)

                        read_noise = get_residual_noise(read_greyd, filter_type='median')
                        read_spect = get_fourier_spectrum(noise_img=read_noise)
                        
                        read_featA = descriptor.get_hog_feature(image=read_greyd, pixel4cell=(96,96), cell4block=(1,1), orientation=8)
                        read_featB = descriptor.get_lbp_ch_feature(image=read_hsvch, bins=265, points=8, radius=1)
                        read_featC = descriptor.get_lbp_ch_feature(image=read_ycrcb, bins=265, points=8, radius=1)
                        read_featD = descriptor.get_glcm_feature(image=read_spect, dists=[1,2], shades=20)
                        
                        read_feats = np.concatenate((read_featA, read_featB, read_featC, read_featD), axis=0)
                        read_spect = (read_spect / np.max(read_spect)) * 255
                        
                        if saveCopy:
                            spec_video.write(read_spect.astype('uint8'))
                            tiny_video.write(np.hstack((read_hsvch, read_ycrcb)))
                        if show:
                            cv.imshow('face', read_color)
                            cv.imshow('spec', cv.normalize(read_spect, 0, 255, cv.NORM_MINMAX))
                            cv.waitKey(1)
                        if not np.any(np.isnan(read_feats)):
                            feature_list.append(read_feats)
                            label_list.append(label)
                            path_list.append(path)
                else:
                    break
                frame_counter += 1

            if verbose:
                print(overall_counter + 1, inner_counter + 1, path, label, len(read_featA), len(read_featB), len(read_featC), len(read_featD))
            if inner_counter % 100 == 0:
                np.save(file_name, [feature_list, label_list, path_list])
            inner_counter += 1
        else:
            if verbose:
                print(overall_counter + 1, inner_counter + 1, path, label, 'WARNING: feature previously extracted!')
        overall_counter += 1
    np.save(file_name, [feature_list, label_list, path_list])
    read_video.release()
    if saveCopy:
        spec_video.release()
        tiny_video.release()

def main():
    # Handle arguments
    parser = argparse.ArgumentParser(description='Extracting Features from Dataset')
    parser.add_argument('-s', '--drop_frame', help='Define number of skipped frames', required=False, default=05, type=str)
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default=os.path.join(HOME, "REMOTE/DATASETS/TEMP/CASIA-FASD"), type=str)
    parser.add_argument('-m', '--mode_exec', help='Choose to extract feature from Train or Test files', required=False, default='None', type=str)

    parser.add_argument('-te', '--testing_file',  help='Path to testing txt file',  required=False, default=os.path.join(HOME, "REMOTE/DATASETS/TEMP/CASIA-FASD/videos_test.txt"), type=str)
    parser.add_argument('-tr', '--training_file', help='Path to training txt file', required=False, default=os.path.join(HOME, "REMOTE/DATASETS/TEMP/CASIA-FASD/videos_train.txt"), type=str)

    # Storing in variables
    args = parser.parse_args()
    DROP_FRAME = int(args.drop_frame)
    FOLDER_PATH = str(args.folder_path)
    MODE_EXEC = str(args.mode_exec).lower()
    
    TEST_FILE = str(args.testing_file)
    TRAIN_FILE = str(args.training_file)

    # Split dataset into train and test sets    
    test_set = load_txt_file(file_name=TEST_FILE)
    train_set = load_txt_file(file_name=TRAIN_FILE)

    if MODE_EXEC == 'train':
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=train_set, frame_drop=DROP_FRAME, scale=1.0, file_name='CASIA-train.npy', verbose=True)
    elif MODE_EXEC == 'test':
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=test_set, frame_drop=DROP_FRAME, scale=1.0, file_name='CASIA-test.npy', verbose=True)
    elif MODE_EXEC == 'none':
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=train_set, frame_drop=DROP_FRAME, scale=1.0, file_name='CASIA-train.npy', verbose=True)
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=test_set, frame_drop=DROP_FRAME, scale=1.0, file_name='CASIA-test.npy', verbose=True)

if __name__ == "__main__":
    main()
