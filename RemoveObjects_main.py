import glob
from tkinter import Image
import time
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sewar.full_ref import mse,ssim, rmse
import RemoveObjects_process as ro


def MSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1]  # img1 and 2 should have same shape
    err = summed / num_pix
    return err

if __name__ == '__main__':

    debug = False
    imgs = []
    #path = "TestImages/Dataset05/"
    path = "TestImages/Dataset09/"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for filename in sorted(glob.glob(path+'*.jpg')):  # assuming gif
        im = cv2.imread(filename)
        imgs.append(im)
    print(len(imgs))

    gt_img = cv2.imread("TestImages/Dataset09/gt/gt01.jpg")
    imgs.append(gt_img)
    start_time = time.time()
    trans_imgs=ro.alignment(imgs)
    alignment_time = (time.time() - start_time)
    print("--- %s seconds ---" % alignment_time)

   # id= 0
    #for img in trans_imgs:
     #   cv2.imwrite(("RO_data/aligned_image_" + str(id) + ".jpg"), img)
     #   id+=1

    final_img=ro.obj_remover(trans_imgs[:-1])
    end_time = (time.time() - start_time)
    print("--- %s seconds ---" % end_time)
    print("pomer %s" % ((alignment_time/end_time)*100))
#    cv2.imwrite((path + "final_img.jpg"), final_img)

    cv2.imshow("gt_img",gt_img)
    cv2.imshow("final_img",final_img)
    cv2.waitKey(0)
    gt_img_mask=cv2.threshold(cv2.cvtColor(trans_imgs[-1], cv2.COLOR_BGR2GRAY), 5, 254, cv2.THRESH_BINARY)[1]
    final_img_mask = cv2.threshold(cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY), 5, 254, cv2.THRESH_BINARY)[1]
    img_mask = cv2.bitwise_and(gt_img_mask,final_img_mask)
    #
    crop_final_img = cv2.bitwise_and(final_img,final_img,mask=img_mask)
    crop_gt_img = cv2.bitwise_and(trans_imgs[-1],trans_imgs[-1],mask = img_mask)
    #
    # gt_img=cv2.imread("TestImages/Dataset09/gt/gt01.jpg")
    # cv2.imshow("gt_img",crop_gt_img)
    # #cv2.imwrite("gt_img_crop.jpg", crop_gt_img)
    # cv2.imshow("final_img",crop_final_img)
    #
    print("mse hodnota: ",MSE(crop_final_img,crop_gt_img))
    # print(crop_final_img.shape)
    # cv2.waitKey(0)
    # final_img = imutils.resize(final_img, width=1000)
    # cv2.imshow("Background subs", final_img)
    # cv2.waitKey(0)
