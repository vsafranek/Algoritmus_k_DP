import glob
from tkinter import Image
from PIL import Image,ImageFilter
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt

import ReturnTime_process
import random


if __name__ == '__main__':

    debug = True
    imgs = []
    #path = "TestImages/Dataset05/"
    #path = "ReturnTime/Dataset02/"
    path = "TestImages/"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for filename in sorted(glob.glob(path+'*.jpg')):  # assuming gif
        im = cv2.imread(filename)
        imgs.append(im)
    print(len(imgs))
    rand=random.randint(0,len(imgs)-1)
    print("vyber obrazku: ",rand)

    ksize=(15,15)
    imgs[rand] =cv2.blur(imgs[rand],ksize)
    #cv2.imshow("blured img", imgs[rand])
    #cv2.waitKey(0)
    i=0
    while i <len(imgs)-1:
        print(i)
        choosed_imgs=[imgs[i],imgs[i+1]]
        ReturnTime_process.image_detector(choosed_imgs,"orb")
        i=i+1


