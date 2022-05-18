import glob
from tkinter import Image

import imutils
import numpy as np
import cv2
import Composition_process



if __name__ == '__main__':

    white_board = cv2.imread("white_board.jpg")

    image = cv2.cvtColor(white_board, cv2.COLOR_BGR2GRAY)
    center = [round(image.shape[0] / 2), round(image.shape[1] / 2)]


    image = image / (image[center[0], center[1]])
    ones = np.ones(image.shape)

    intensity = 1
    intens_image=ones+((ones-image) * intensity)
    max_bright=np.max(intens_image)
    intens_image = intens_image/max_bright
    intens_image = cv2.blur(intens_image, (20,20))
    #intens_image = ones - intens_image
    white_mask = max_bright*0.95*intens_image#.astype('uint8')


    #cv2.imshow("maska",image)
    #cv2.imshow("intense",intens_image)
    #cv2.waitKey(0)
    debug = True
    imgs = []
    #path = "TestImages/Dataset05/"
    path = "CompImages/Dataset01/"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for filename in sorted(glob.glob(path+'*.jpg')):  # assuming gif
        im = cv2.imread(filename)
        #white_mask= cv2.resize(white_mask, (im.shape[1],im.shape[0]), interpolation=cv2.INTER_AREA)
        for i in range(3):
            pass
            im[:, :, i] = im[:, :, i]*white_mask
        imgs.append(im)
    print(len(imgs))

    new_imgs=[]
    for index in [0,3]:
        crop_image=cv2.imread("SC_data/crop_image.jpg")
        new_imgs.append(imgs[index])
        #new_imgs.append(crop_image)

    final_img=Composition_process.panorama_orb(new_imgs,"brisk")


    #cv2.imshow("Final img", final_img)
    #cv2.imwrite("SC_data/Final img_01.jpg",final_img)

    cv2.waitKey(0)