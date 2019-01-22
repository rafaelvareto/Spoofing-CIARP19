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

def obtain_video_features(folder_path, dataset_tuple, frame_drop=1, size=(400,300), file_name='video_features.npy', saveCopy=False, show=False, verbose=False):
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
            if verbose:
                print(overall_counter + 1, inner_counter + 1, path, label)
            frame_counter = 0
            probe_fourcc = cv.VideoWriter_fourcc(*'MP42') 
            read_path = os.path.join(folder_path, path)
            read_video = cv.VideoCapture(read_path)
            if saveCopy:
                spec_video = cv.VideoWriter(read_path.replace('.mov', '_spec.avi'), probe_fourcc, 20.0, size, isColor=False)
                tiny_video = cv.VideoWriter(read_path.replace('.mov', '_tiny.avi'), probe_fourcc, 20.0, size, isColor=True)
            while(read_video.isOpened()):
                ret, read_frame = read_video.read()
                if ret:
                    if frame_counter % frame_drop == 0:
                        read_color = cv.resize(read_frame, (size[0], size[1]), interpolation=cv.INTER_AREA)
                        read_greyd = cv.cvtColor(read_color, cv.COLOR_BGR2GRAY)
                        read_noise = get_residual_noise(read_greyd, filter_type='median')
                        read_spect = get_fourier_spectrum(noise_img=read_noise)
                        read_featA = descriptor.get_hog_feature(image=read_greyd, pixel4cell=(64,64), cell4block=(1,1), orientation=8)
                        read_featB = descriptor.get_hog_feature(image=read_spect, pixel4cell=(64,64), cell4block=(1,1), orientation=8)
                        read_featC = descriptor.get_glcm_feature(image=read_spect, dists=[1,2], shades=20)
                        read_feats = np.concatenate((read_featA, read_featB, read_featC), axis=0)
                        read_spect = (read_spect / np.max(read_spect)) * 255
                        if saveCopy:
                            spec_video.write(read_spect.astype('uint8'))
                            tiny_video.write(read_color)
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
            inner_counter += 1
            if inner_counter % 100 == 0:
                np.save(file_name, [feature_list, label_list, path_list])
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
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release"), type=str)
    parser.add_argument('-m', '--mode_exec', help='Choose to extract feature from Train or Test files', required=False, default='None', type=str)

    parser.add_argument('-te', '--testing_file', help='Path to testing txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/test_videos.txt"), type=str)
    parser.add_argument('-tr', '--training_file', help='Path to training txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/train_videos.txt"), type=str)

    # Storing in variables
    args = parser.parse_args()
    FOLDER_PATH = str(args.folder_path)
    MODE_EXEC = str(args.mode_exec).lower()
    TEST_FILE = str(args.testing_file)
    TRAIN_FILE = str(args.training_file)

    # Split dataset into train and test sets
    test_set = load_txt_file(file_name=TEST_FILE)
    train_set = load_txt_file(file_name=TRAIN_FILE)

    if MODE_EXEC == 'train':
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=train_set, frame_drop=1, size=(400,300), file_name='SiW-train-new.npy', verbose=True)
    elif MODE_EXEC == 'test':
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=test_set, frame_drop=1, size=(400,300), file_name='SiW-test-new.npy', verbose=True)
    elif MODE_EXEC == 'none':
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=train_set, frame_drop=1, size=(400,300), file_name='SiW-train-new.npy', verbose=True)
        obtain_video_features(folder_path=FOLDER_PATH, dataset_tuple=test_set, frame_drop=1, size=(400,300), file_name='SiW-test-new.npy', verbose=True)

if __name__ == "__main__":
    main()
