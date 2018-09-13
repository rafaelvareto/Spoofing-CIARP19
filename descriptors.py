'''
https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html
https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
https://www.ucalgary.ca/mhallbey/tutorial-glcm-texture
https://www.uio.no/studier/emner/matnat/ifi/INF4300/h08/undervisningsmateriale/glcm.pdf
'''

import cv2 as cv
import numpy as np
import skimage as ski

from skimage.feature import greycomatrix, greycoprops, hog, local_binary_pattern

class Descriptors:
    def __init__(self):
        # Instantiation of OpenCV methods
        self._hog = cv.HOGDescriptor(_winSize=(32,32), _blockSize=(32,32), _blockStride=(8,8), _cellSize=(16,16), _nbins=9)
        self._orb = cv.ORB_create()
        self._sift = cv.xfeatures2d.SIFT_create()
        self._surf = cv.xfeatures2d.SURF_create()

    def __normalize(self, vector):
        return vector / np.sum(vector)

    def convert_to_int(self, image):
        return image.astype(np.uint8)

    def get_akaze_feature(self, image):
        int_image = self.convert_to_int(image)
        pass

    def get_brief_feature(self, image):
        int_image = self.convert_to_int(image)
        pass

    def get_brisk_feature(self, image):
        int_image = self.convert_to_int(image)
        pass

    def get_daisy_feature(self, image):
        int_image = self.convert_to_int(image)
        feats = ski.feature.daisy(int_image, step=90, radius=30, rings=2, histograms=6, orientations=8)
        return feats[0]

    def get_fast_feature(self, image):
        int_image = self.convert_to_int(image)
        pass

    def get_freak_feature(self, image):
        int_image = self.convert_to_int(image)
        pass

    def get_glcm_feature(self, image, dists=[1,2], shades=16, stats=['contrast','dissimilarity','homogeneity','energy','correlation','ASM']):
        '''
        Usually the GLCM texture features are from gray scale images of 256 gray levels in thee main degrees Horizontal, Vertical, and diagonal (0, 45,90,135).
        '''
        feats = list()
        int_image = self.convert_to_int(image)
        matrix = ski.feature.greycomatrix(int_image, distances=dists, angles=[np.deg2rad(0), np.deg2rad(45), np.deg2rad(90), np.deg2rad(135)], levels=shades, normed=True)
        for item in stats:
            feature = ski.feature.greycoprops(matrix, item).flatten()
            feats.extend(self.__normalize(feature))
        return feats
        
    def get_hog_feature(self, image, pixel4cell=(32,32), cell4block=(2,2), orientation=9, version='ski'):
        int_image = self.convert_to_int(image)
        if version == 'cv':
            feats = self._hog.compute(int_image)
            feats = np.array([float(item) for item in feats])
        elif version == 'ski':
            feats = ski.feature.hog(int_image, pixels_per_cell=pixel4cell, cells_per_block=cell4block, orientations=orientation, block_norm='L2')
        else:
            raise ValueError('ERROR: Two HOG methods available: ski (skimage) and cv (opencv)')
        return feats

    def get_kaze_feature(self, image):
        int_image = self.convert_to_int(image)
        pass

    def get_lbp_feature(self, image, bins=265):
        int_image = self.convert_to_int(image)
        feats = ski.feature.local_binary_pattern(int_image, P=24, R=8)
        hist = np.histogram(feats, normed=True, bins=bins)
        feature = hist[0].flatten()
        return self.__normalize(feature)

    def get_orb_feature(self, image, kpoints=100):
        int_image = self.convert_to_int(image)
        keypoints = self._orb.detect(int_image, None)
        keys, feats = self._orb.compute(int_image, keypoints[0:kpoints])
        return feats.flatten()

    def get_sift_feature(self, image, kpoints=100):
        int_image = self.convert_to_int(image)
        keypoints = self._sift.detect(int_image)
        keys, feats = self._sift.compute(int_image, keypoints[0:kpoints])
        return feats.flatten()

    def get_surf_feature(self, image, kpoints=100):
        int_image = self.convert_to_int(image)
        keypoints = self._surf.detect(int_image)
        keys, feats = self._surf.compute(int_image, keypoints[0:kpoints])
        return feats.flatten()

