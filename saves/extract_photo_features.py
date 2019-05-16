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

def obtain_image_features(folder_path, size=(640, 360), file_name='image_features.npy', saveCopy=False, show=False, verbose=False):
    descriptor = Descriptors()
    feature_list = list()
    label_list = list()
    path_list = list()

    dir_list = [dir_item for _root, _path, _file in os.walk(folder_path) for dir_item in _file if dir_item.endswith('.jpg')]
    dir_list.sort()
    name_list = [os.path.splitext(dir_item.replace('_face', ''))[0] for dir_item in dir_list]

    if os.path.isfile(file_name):
        print('>> Importing', file_name)
        feature_list, label_list, path_list = np.load(file_name)
        feature_list = list(feature_list)
        label_list = list(label_list)
        path_list = list(path_list)

    inner_counter = overall_counter = 0
    for (dir_item, name_item) in zip(dir_list, name_list):
        if (dir_item not in path_list):
            if verbose:
                print(dir_item, name_item)
            sample_path = os.path.join(folder_path, dir_item)
            sample_image = cv.imread(sample_path, cv.IMREAD_COLOR)
            scaled_image = cv.resize(sample_image, (size[0], size[1]), interpolation=cv.INTER_AREA)

            scaled_greyd = cv.cvtColor(scaled_image, cv.COLOR_BGR2GRAY)
            scaled_hsvch = cv.cvtColor(scaled_image, cv.COLOR_BGR2HSV)
            scaled_ycrcb = cv.cvtColor(scaled_image, cv.COLOR_BGR2YCrCb)

            sample_noise = get_residual_noise(scaled_greyd, filter_type='median')
            sample_spect = get_fourier_spectrum(noise_img=sample_noise)
            
            sample_featA = descriptor.get_hog_feature(image=scaled_greyd, pixel4cell=(96,96), cell4block=(1,1), orientation=8)
            sample_featB = descriptor.get_lbp_ch_feature(image=scaled_hsvch, bins=265, points=8, radius=1)
            sample_featC = descriptor.get_lbp_ch_feature(image=scaled_ycrcb, bins=265, points=8, radius=1)
            sample_featD = descriptor.get_glcm_feature(image=sample_spect, dists=[1,2], shades=20)
            
            sample_feats = np.concatenate((sample_featA, sample_featB, sample_featC, sample_featD), axis=0)
            sample_spect = (sample_spect / np.max(sample_spect)) * 255

            if verbose:
                print(overall_counter + 1, inner_counter + 1, dir_item, name_item, len(sample_featA), len(sample_featB), len(sample_featC), len(sample_featD))
            if saveCopy:
                temp_name01 = dir_item
                temp_name02 = dir_item
                cv.imwrite(os.path.join(folder_path, temp_name01.replace('.jpg','_tiny.png')), scaled_hsvch)
                cv.imwrite(os.path.join(folder_path, temp_name02.replace('.jpg','_spec.png')), sample_spect.astype('uint8'))
                print(dir_item, temp_name01, temp_name02)
            if show:
                cv.imshow('image', scaled_image)
                cv.imshow('spec', cv.normalize(sample_spect, 0, 255, cv.NORM_MINMAX))
                cv.waitKey(1)
            if not np.any(np.isnan(sample_feats)):
                feature_list.append(sample_feats)
                label_list.append(name_item)
                path_list.append(dir_item)
            if inner_counter % 20 == 0:
                np.save(file_name, [feature_list, label_list, path_list])
            inner_counter += 1
        else:
            if verbose:
                print(overall_counter + 1, inner_counter + 1, dir_item, name_item, 'WARNING: feature previously extracted!')
        overall_counter += 1
    np.save(file_name, [feature_list, label_list, path_list])

def main():
    # Handle arguments
    parser = argparse.ArgumentParser(description='Extracting Features from folder containing images')
    parser.add_argument('-f', '--folder_path', help='Path to image folder', required=False, default=os.path.join(HOME, "GIT/Spoofing-ICASSP19/datasets/WAX"), type=str)

    # Storing in variables
    args = parser.parse_args()
    FOLDER_PATH = str(args.folder_path)
    obtain_image_features(folder_path=FOLDER_PATH, size=(680, 520), file_name='WAX-feats.npy', saveCopy=True, show=True, verbose=True)


if __name__ == "__main__":
    main()
