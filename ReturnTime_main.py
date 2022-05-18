import glob
import time
from tkinter import Image
from PIL import Image,ImageFilter
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt

import ReturnTime_process
import random


def F1_score(imgs, filter):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for id in range(0, len(imgs)):
        if imgs[id] == 1 and filter[id] == 1:
            tp += 1
        elif imgs[id] == 1 and filter[id] == 0:
            fn += 1
        elif imgs[id] == 0 and filter[id] == 0:
            tn += 1
        elif imgs[id] == 0 and filter[id] == 1:
            fp += 1
    precission = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precission * recall) / (precission + recall)
    return F1


if __name__ == '__main__':

    debug = True
    imgs = []
    #path = "TestImages/Dataset05/"
    path = "ReturnTime/Dataset04/"
    #path = "TestImages/Dataset04/"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for filename in sorted(glob.glob(path+'*.jpeg')):  # assuming gif
        im = cv2.imread(filename)
        #cv2.imshow("img",im)
        #cv2.waitKey(0)
        imgs.append(im)
    print(len(imgs))

    blured_imgs = [1,0,0,1,0,1,1,1,0,1,1,0,1,1,1,0,0,1,0,1,1,1,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1]
    repeated_imgs = [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    ksize = [(15, 15), (10, 10), (20, 20), (5,5)]
    for id in range(0,10):
        rand=random.randint(0,len(imgs)-1)
        krand = random.randint(0,len(ksize)-1)
        #imgs[rand] =cv2.blur(imgs[rand],ksize[3])
        #filename=str("ReturnTime/Dataset03/blured_img_"+str(id)+"_.jpg")
        #cv2.imwrite(filename,imgs[rand])
        #print("rozmazany obrazek: ", rand, " kernelem: ", krand)
        #cv2.imshow(str(id), imgs[rand])
    #cv2.waitKey(0)
    bfilter = True
    rfilter = True
    i=0
    kpt_data=[]
    kpt_match_data=[]
    kpt_good_match_data=[]
    while i <len(imgs)-1:
        print("obrazek: ",i+1, " a ",i+2)
        scale_percent=5
        width = 400#int(imgs[i].shape[1] * scale_percent / 100)
        height = int(imgs[i].shape[0] * width/imgs[i].shape[1])
        dim = (width, height)

        img_small1=cv2.resize(imgs[i],dim,interpolation=cv2.INTER_CUBIC)
        img_small2=cv2.resize(imgs[i+1],dim,interpolation=cv2.INTER_CUBIC)
        choosed_imgs=[img_small1,img_small2]
        [kpt1, kpt2, kptM, good]=ReturnTime_process.image_detector(choosed_imgs,"orb")
        kpt_data.append(kpt1)
        kpt_match_data.append(kptM)
        kpt_good_match_data.append(good)
        i=i+1
    if bfilter==True:
        bfilter = ReturnTime_process.blurred_filter(kpt_data,kpt_match_data)
        bfilter.append(1)
        print(np.sum(bfilter))
        print("GT:   ",blured_imgs)
        print("auto: ",bfilter)
        F1=F1_score(blured_imgs[:(len(bfilter)-1)],bfilter)
        print(F1)
    else:
        bfilter = np.ones(len(imgs),dtype=int)
    if rfilter==True:
        rfilter = ReturnTime_process.repeating_filter(imgs,kpt_good_match_data)
        rfilter.append(1)
        F1 = F1_score(repeated_imgs, rfilter)
        print(F1)

    else:
        rfilter = np.ones(len(imgs), dtype=int)
    filter=[]

    for i in range(0,len(bfilter)):
        if bfilter[i] == 1 and rfilter[i] == 1:
            filter.append(1)
            #cv2.imshow(str(i),imgs[i]
            #print(kpt_data[i])
        else:
            filter.append(0)
    print("final filter: ",filter)
    for item in range(0,len(filter)):
        if filter[item] == 1:
            cv2.imshow(str(item),imgs[item])
            cv2.waitKey(0)

    #return ("".join(filter))

