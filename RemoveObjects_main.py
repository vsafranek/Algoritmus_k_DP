import glob
from tkinter import Image

import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import RemoveObjects_process as ro
import image_match



if __name__ == '__main__':

    debug = True
    imgs = []
    #path = "TestImages/Dataset05/"
    path = "CompImages/"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for filename in glob.glob(path+'*.jpeg'):  # assuming gif
        im = cv2.imread(filename)
        imgs.append(im)
    print(len(imgs))

    trans_imgs=ro.alignment(imgs)

    if debug:
        plt.imshow(trans_imgs[0])
        plt.show()
        plt.imshow(trans_imgs[1])
        plt.show()
        plt.imshow(trans_imgs[2])
        plt.show()
        plt.imshow(trans_imgs[3])
        plt.show()
        plt.imshow(trans_imgs[4])
        plt.show()

    final_img=ro.obj_remover(trans_imgs)
    final_img = imutils.resize(final_img, width=1000)
    cv2.imshow("Background subs", final_img)
    cv2.waitKey(0)