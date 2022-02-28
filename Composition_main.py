import glob
from tkinter import Image

import imutils
import numpy as np
import cv2
import image_match
import Composition_process



if __name__ == '__main__':

    debug = True
    imgs = []
    #path = "TestImages/Dataset05/"
    path = "CompImages/Dataset_03/"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for filename in sorted(glob.glob(path+'*.jpeg')):  # assuming gif
        im = cv2.imread(filename)
        imgs.append(im)
    print(len(imgs))

    final_img=Composition_process.panorama_orb(imgs,"sift")


    cv2.imshow("Background subs", final_img)
    cv2.waitKey(0)