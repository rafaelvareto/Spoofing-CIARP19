import cv2 as cv
import numpy as np 

from face_spoofing import FaceSpoofing


class Video:
    def __init__(self):
        print('Video Player Class')
        self._input_name = None
        self._output_name = None
        self._stop = False
        self._videoCapture = cv.VideoCapture()

    def set_input_video(self, in_name):
        self._input_name = in_name
        self._videoCapture = cv.VideoCapture(self._input_name)

    def set_output_video(self, out_name):
        self._output_name = out_name

    def get_frames_per_second(self):
        return self._videoCapture.get(cv.CAP_PROP_FPS)

    def get_position_millisecond(self):
        return self._videoCapture.get(cv.CAP_PROP_POS_MSEC)

    def get_position_frame_number(self):
        return self._videoCapture.get(cv.CAP_PROP_POS_FRAMES)
    
    def get_position_frame_size(self):
        height = self._videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = self._videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)
        return (height, width)
    
    def get_total_frame_count(self):
        return self._videoCapture.get(cv.CAP_PROP_FRAME_COUNT)

    def get_video_codec(self):
        return self._videoCapture.get(cv.CAP_PROP_FOURCC)

    def play(self, spoofer=None, frame_pos=0, delay=1):
        self._videoCapture.set(cv.CAP_PROP_POS_FRAMES, frame_pos)
        while(self._videoCapture.isOpened()):
            ret, frame = self._videoCapture.read()
            if ret:
                if spoofer:
                    gray_img = spoofer.get_gray_image(frame)
                    noise_img = spoofer.get_residual_noise(gray_img, filter_type='median')
                    spec_img = spoofer.get_fourier_spectrum(noise_img)
                    spec_feat = spoofer._descriptor.get_glcm_feature(spec_img)
                    cv.imshow('spectrum', cv.normalize(spec_img, 0, 255, cv.NORM_MINMAX))
                    cv.waitKey(delay)
                else:
                    cv.imshow('color', frame)
                    cv.waitKey(delay)
            else:
                break
        cv.destroyAllWindows()