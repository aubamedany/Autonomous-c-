"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)
        return out_img

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def show_curvature(input_path):
    calibration = CameraCalibration('camera_cal', 9, 6)
    thresholding = Thresholding()
    transform = PerspectiveTransformation()
    lanelines = LaneLines()
    img = mpimg.imread(input_path)
    out_img = np.copy(img)
    img = calibration.undistort(img)
    img = transform.forward(img)
    bird_eye_img = np.copy(img)
    img = thresholding.forward(img)
    img = lanelines.forward(img)
    img = transform.backward(img)
    print(lanelines.measure_curvature())
def main():

    calibration = CameraCalibration('camera_cal', 9, 6)
    thresholding = Thresholding()
    transform = PerspectiveTransformation()
    lanelines = LaneLines()
    video = cv2.VideoCapture("/Users/namle/BachKhoa/HK232/DOANAI/Advanced-Lane-Lines/project_video.mp4")

    video.set(cv2.CAP_PROP_FPS, 10)
    while (video.isOpened()):
        try:
            flag, img = video.read()
            img_real = np.copy(img)
            if not flag: break
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            out_img = np.copy(img)
            img = calibration.undistort(img)
            img = transform.forward(img)
            bird_eye_img = np.copy(img)
            img = thresholding.forward(img)
            img = lanelines.forward(img)
            img = transform.backward(img)
            out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
            out_img = lanelines.plot(out_img)

            combined_img = np.hstack((out_img, img_real,bird_eye_img))
            cv2.imshow("Combined Images", combined_img)
            if cv2.waitKey(1) == ord('q'):break
        except Exception as e:
            print(e)
   
show_curvature("/Users/namle/BachKhoa/HK232/DOANAI/Advanced-Lane-Lines/test_images/challenge_video_frame_10.jpg")
# if __name__ == "__main__":

#     main()
