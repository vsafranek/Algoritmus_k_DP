import time

import cv2
import numpy
import numpy as np
from scipy.ndimage import laplace
from matplotlib import pyplot as plt


def panorama_orb(data, detector_type, ):
    debug = True
    debug2 = False
    debug3= False
    grand_truth = False

    width = int( data[0].shape[1])
    height = int( data[0].shape[0])
    imgs=[]

    width_small = 800
    height_small = int(height * width_small / width)
    dim_small = (width_small, height_small)
    id=0
    for item in data:
        small_image=cv2.resize(item, dim_small, interpolation=cv2.INTER_AREA)
        cv2.imwrite((str(id)+".jpg"),small_image)
        imgs.append(small_image)
        id+=1

    w = int(2 * width_small)
    h = int(1 * height_small)

    wm= int(3.5 * width_small)
    hm= int(2.4 * height_small)

    first_run_done=False
    #transformed_imgs = [None] * len(imgs)
    template = imgs[0]
    prim_image_mask = np.zeros(template.shape, dtype="uint8")
    prim_image_mask[:,:] = 255


    #prim_image_mask = cv2.copyMakeBorder(prim_image_mask, 2000, 2032, 3024, 3024, cv2.BORDER_CONSTANT, value=(000, 000, 000))
    #moved_template=cv2.copyMakeBorder(template,2000,2032,3024,3024,cv2.BORDER_CONSTANT,value=(000,000,000))
    position = "BOTTOM_LEFT"

    if position == "TOP_LEFT":
        w_pos=int(w-template.shape[1])
        h_pos=int(h-template.shape[0])
        print(h_pos)
        prim_image_mask = cv2.copyMakeBorder(prim_image_mask, 0, h_pos, 0, w_pos, cv2.BORDER_CONSTANT,
                                         value=(000, 000, 000))
        moved_template = cv2.copyMakeBorder(template, 0, h_pos, 0, w_pos, cv2.BORDER_CONSTANT, value=(000, 000, 000))
    elif position == "MIDDLE":
        w=wm
        h=hm
        w_pos=int((w - template.shape[1]) / 2)
        h_pos=int((h-template.shape[0]) / 2)
        prim_image_mask = cv2.copyMakeBorder(prim_image_mask,h_pos,h_pos,w_pos,w_pos,cv2.BORDER_CONSTANT,value=(000,000,000))
        moved_template = cv2.copyMakeBorder(template, h_pos, h_pos, w_pos, w_pos, cv2.BORDER_CONSTANT,
                                             value=(000, 000, 000))
    elif position == "TOP_MIDDLE":
        w = wm
        h = hm
        w_pos = int((w - template.shape[1]) / 2)
        h_pos = int(h - template.shape[0])
        prim_image_mask = cv2.copyMakeBorder(prim_image_mask, 0, h_pos, w_pos, w_pos, cv2.BORDER_CONSTANT,
                                             value=(000, 000, 000))
        moved_template = cv2.copyMakeBorder(template, 0, h_pos, w_pos, w_pos, cv2.BORDER_CONSTANT,
                                            value=(000, 000, 000))
    elif position == "BOTTOM_MIDDLE":
        w = wm
        h = hm
        w_pos = int((w - template.shape[1]) / 2)
        h_pos = int(h - template.shape[0])
        prim_image_mask = cv2.copyMakeBorder(prim_image_mask, h_pos,0, w_pos, w_pos, cv2.BORDER_CONSTANT,
                                             value=(000, 000, 000))
        moved_template = cv2.copyMakeBorder(template, h_pos,0, w_pos, w_pos, cv2.BORDER_CONSTANT,
                                            value=(000, 000, 000))
    elif position == "BOTTOM_LEFT":
        w_pos = int(w - template.shape[1])
        h_pos = int(h - template.shape[0])
        prim_image_mask = cv2.copyMakeBorder(prim_image_mask, h_pos, 0, 0, w_pos,  cv2.BORDER_CONSTANT,
                                             value=(000, 000, 000))
        moved_template = cv2.copyMakeBorder(template, h_pos, 0, 0, w_pos, cv2.BORDER_CONSTANT, value=(000, 000, 000))
    elif position == "TOP_RIGHT":
        w_pos = int(w - template.shape[1])
        h_pos = int(h - template.shape[0])
        prim_image_mask = cv2.copyMakeBorder(prim_image_mask, 0, h_pos, w_pos, 0, cv2.BORDER_CONSTANT,
                                             value=(000, 000, 000))
        moved_template = cv2.copyMakeBorder(template, 0, h_pos, w_pos,0, cv2.BORDER_CONSTANT, value=(000, 000, 000))
    elif position == "BOTTOM_RIGHT":
        w_pos = int(w - template.shape[1])
        h_pos = int(h - template.shape[0])
        prim_image_mask = cv2.copyMakeBorder(prim_image_mask, h_pos, 0, w_pos, 0, cv2.BORDER_CONSTANT,
                                             value=(000, 000, 000))
        moved_template = cv2.copyMakeBorder(template, h_pos, 0,  w_pos, 0, cv2.BORDER_CONSTANT, value=(000, 000, 000))

    print("width",w)
    print("height", h)

    #cv2.imshow("moved template",moved_template)
    #cv2.imwrite("SC_data/moved_templee.jpg",moved_template)
    #cv2.waitKey(0)

    start_time=time.time()
    template=np.copy(moved_template)
    #transformed_imgs[0] = template
    if detector_type == "akaze":
        detector = cv2.AKAZE_create()
    elif detector_type == "brisk":
        detector = cv2.BRISK_create()
    elif detector_type == "orb":
        detector = cv2.ORB_create(10000)
    elif detector_type == "sift":
        keypoint_limit=10000
        detector = cv2.SIFT_create(keypoint_limit)
    elif detector_type =="surf":
        keypoint_limit = 3000
        detector = cv2.xfeatures2d.SURF_create(keypoint_limit)
    else:
        detector = cv2.ORB_create(1000)
    kpts = []
    descs = []
    # kpts=[None]*len(imgs)
    # descs=[None]*len(imgs)
    # Create our ORB detector and detect keypoints and descriptors

    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    img = imgs[0]
    i=0
    while i < len(imgs):

        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = detector.detectAndCompute(template, None)
        keypoints2, descriptors2 = detector.detectAndCompute(imgs[i], None)
        print("pocet nalezených keypointů: ", len(keypoints1), len(keypoints2))
        # It will find all of the matching keypoints on two images
        if detector_type in ["sift"]:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = matcher.match(descriptors1, descriptors2, None)
            matches = sorted(matches, key=lambda x: x.distance)
            keep = int(len(matches) * 0.4)
            best_matches = matches[:500]


        elif detector_type in ["orb", "akaze", "brisk","gt"]:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(descriptors1, descriptors2, None)
            matches = sorted(matches, key=lambda x: x.distance)
            keep = int(len(matches) * 0.4)
            best_matches = matches[:500]
        # best_matches = []
        # for m, n in matches:
        #    if m.distance < 0.8 * n.distance:
        #       best_matches.append(m)

        no_of_matches = len(best_matches)
        print("best matches", no_of_matches)
        # Set minimum match condition
        MIN_MATCH_COUNT = 100

        if debug2:
            cv2.imwrite(("SC_data/"+detector_type+"_matches.jpg"), draw_matches(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), keypoints1,
                                               cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY), keypoints2,
                                               best_matches[:30]))
            cv2.imwrite(("SC_data/"+detector_type+"_template.jpg"),
                       cv2.drawKeypoints(cv2.cvtColor(template,cv2.COLOR_BGR2GRAY), [keypoints1[m.queryIdx] for m in best_matches], None, (255, 0, 255)))
            cv2.imwrite(("SC_data/"+detector_type+"_imgs[s].jpg"),
                       cv2.drawKeypoints(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY), [keypoints2[m.trainIdx] for m in best_matches], None, (255, 0, 255)))
            cv2.waitKey(0)
        if len(best_matches) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography

            if grand_truth:
                if i == 1:
                    src_array = []
                    src_array.append((528,333))
                    src_array.append((617,352))
                    src_array.append((628,480))
                    src_array.append((534,476))

                    dst_array = []
                    dst_array.append((160, 319))
                    dst_array.append((246, 353))
                    dst_array.append((250, 480))
                    dst_array.append((159, 468))

                    src_pts = np.float32(src_array).reshape(-1, 1, 2)
                    dst_pts = np.float32(dst_array).reshape(-1, 1, 2)
                else:
                    src_pts = np.float32([keypoints1[m.queryIdx].pt for (j, m) in enumerate(best_matches)]).reshape(-1,1,2)
                    dst_pts = np.float32([keypoints2[m.trainIdx].pt for (j, m) in enumerate(best_matches)]).reshape(-1,1,2)
            else:

                # num=0
                #for (j, m) in enumerate(best_matches):
                #   if keypoints1[m.trainIdx].pt[1] > 430:
                #      num+=1
                #print(num)
                src_pts = np.float32([keypoints1[m.queryIdx].pt for (j, m) in enumerate(best_matches)]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for (j, m) in enumerate(best_matches)]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            # dst=warpImages(template,imgs[i],homography)
            dst = cv2.warpPerspective(imgs[i], homography, (w, h),flags=cv2.INTER_NEAREST)

            #cv2.imshow("dst",dst)
            #cv2.waitKey(0)

            sec_image_mask = np.zeros(imgs[i].shape, dtype=np.uint8)
            sec_image_mask[:, :, :] = 255

            dst_Mask = cv2.warpPerspective(sec_image_mask, homography, (w, h),flags=cv2.INTER_NEAREST)
            if first_run_done:
                prim_image_mask=np.ones(imgs[i].shape, dtype=np.uint8)
                prim_image_mask = np.zeros(dst_Mask.shape, dtype=np.uint8)
                prim_image_mask[0:prim_image_mask_old.shape[0], 0:prim_image_mask_old.shape[1], :] = prim_image_mask_old

                prim_image_mask_old = cv2.bitwise_or(dst_Mask, prim_image_mask)
            else:
                #prim_image_mask = np.zeros(dst_Mask.shape, dtype=np.uint8)
                #prim_image_mask[0:template.shape[0], 0:template.shape[1], :] = 255

                print("size dst_mask", np.shape(dst_Mask))
                print("size prim_image_mask", np.shape(prim_image_mask))
                prim_image_mask_old = cv2.bitwise_or(dst_Mask, prim_image_mask)

            prim_image = np.zeros(dst_Mask.shape, dtype=np.uint8)
            prim_image[0:template.shape[0], 0:template.shape[1], :] = template

            sec_mask = cv2.bitwise_and(cv2.bitwise_or(dst_Mask, prim_image_mask), cv2.bitwise_not(prim_image_mask))



            kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (55,55))
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            prim_mask_erode = cv2.morphologyEx(prim_image_mask, cv2.MORPH_ERODE, kernel7, iterations=1)
            sec_mask_dilate = cv2.morphologyEx(sec_mask, cv2.MORPH_DILATE, kernel7, iterations=1)


            share_mask=cv2.bitwise_and(prim_image_mask,sec_mask_dilate)
            #share_mask= cv2.morphologyEx(share_mask, cv2.MORPH_DILATE, kernel7, iterations=1)

            if np.sum(sec_mask) != 0:
                plus_image = cv2.bitwise_and(dst, dst, mask=sec_mask[:, :, 0])
                image = cv2.bitwise_or(plus_image, prim_image)
                share_image = cv2.bitwise_or(image, image, mask=share_mask[:, :, 0])

                if debug:
                    cv2.imshow("secondary image", dst)
                    cv2.imwrite("SC_data/"+detector_type+"_secondary_image.jpg",dst)
                    cv2.imshow("secondary image mask", dst_Mask)
                    cv2.imshow("primary image mask", prim_image_mask)
                    cv2.imshow(" mask", sec_mask)
                    cv2.imshow("image", image)
                    cv2.imshow("plus image", plus_image)
                    cv2.imshow("share mask",share_mask)
                    cv2.waitKey(0)
                template = np.copy(image)

                share_mask=share_mask[:, :, 0]
                kernel = np.ones((5, 5), np.float32) / 25
                #kernel = np.matrix([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/256
                #kernel = np.matrix([[3,1,-1,-3,-1,1,3],[3,1,-1,-3,-1,1,3],[3,1,-1,-3,-1,1,3],[3,1,-1,-3,-1,1,3],[3,1,-1,-3,-1,1,3]])/15

                kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

                #smoothed_image = cv2.filter2D(src=share_image, ddepth=-1, kernel=kernel)
                image_part1 = cv2.bitwise_and(prim_image,prim_image,mask=share_mask)
                image_part2 = cv2.bitwise_and(dst, dst, mask=share_mask)


                #color = ('b', 'g', 'r')
                #for i, col in enumerate(color):
                #    histr = cv2.calcHist([dst], [i], None, [256], [1, 256])
                #    plt.plot(histr, color=col)
                #    plt.xlim([0, 256])
                #plt.show()
                #cv2.waitKey(0)



                padding = blending(20, 0, prim_image, prim_image_mask, dst, dst_Mask, 0.10)
                padding_num = 1
                if np.sum(padding)<np.sum(plus_image):
                    padding += blending(40, 20, prim_image, prim_image_mask, dst, dst_Mask, 0.20)
                    padding_num = 2
                if np.sum(padding) < np.sum(plus_image):
                    padding += blending(60, 40, prim_image, prim_image_mask, dst, dst_Mask, 0.30)
                    padding_num = 3
                if np.sum(padding) < np.sum(plus_image):
                    padding += blending(80, 60, prim_image, prim_image_mask, dst, dst_Mask, 0.40)
                    padding_num = 4
                if np.sum(padding) < np.sum(plus_image):
                    padding += blending(100, 80, prim_image, prim_image_mask, dst, dst_Mask, 0.50)
                    padding_num = 5
                if np.sum(padding) < np.sum(plus_image):
                    padding += blending(120, 100, prim_image, prim_image_mask, dst, dst_Mask, 0.60)
                    padding_num = 6
                if np.sum(padding) < np.sum(plus_image):
                    padding += blending(140, 120, prim_image, prim_image_mask, dst, dst_Mask, 0.70)
                    padding_num = 7
                if np.sum(padding) < np.sum(plus_image):
                    padding += blending(160, 140, prim_image, prim_image_mask, dst, dst_Mask, 0.80)
                    padding_num = 8
                if np.sum(padding) < np.sum(plus_image):
                    padding += blending(180, 160, prim_image, prim_image_mask, dst, dst_Mask, 0.90)
                    padding_num = 9

                kernel30 = cv2.getStructuringElement(cv2.MORPH_RECT, (padding_num*20, padding_num*20))
                prim_mask_erode = cv2.morphologyEx(prim_image_mask, cv2.MORPH_ERODE, kernel30, iterations=1)
                #padding=padding_part1+padding_part2+padding_part3+padding_part4+padding_part5+padding_part6+padding_part7+padding_part8+padding_part9

                share_mask = cv2.bitwise_and(cv2.bitwise_xor(prim_image_mask, prim_mask_erode), dst_Mask)
                image_holed=cv2.bitwise_or(image,image,mask=cv2.bitwise_not(share_mask[:,:,0]))
                new_image=image_holed+padding
                if debug3:
                    cv2.imshow("padding", padding)
                    cv2.imshow("image border",share_image)
                    cv2.imshow(" new_image",new_image)
                    #cv2.imwrite(("SC_data/"+detector_type+"_new_image.jpg"), new_image)
                    cv2.imshow("image",image)
                    cv2.imshow("hole_image",image_holed)
                    cv2.imshow("plus image", plus_image)
                    cv2.waitKey(0)
                template = np.copy(new_image)
            # dst[0:template.shape[0], 0:template.shape[1]] = template

            # dst = warpImages(template, imgs[i], homography)
            # w = dst.shape[1]
            # h = dst.shape[0]

            # template = np.copy(trim_side(dst, h, w))
            # w = template.shape[1]
            # h = template.shape[0]
            cv2.imshow("final pic", template)
            #cv2.waitKey(0)
            first_run_done=True
            i+=1


    print("doba zpracovani: ",time.time()-start_time)

    image=(trim(crop(template)))
    #cv2.imshow("final pic",image)
    #cv2.imwrite(("SC_data/"+detector_type+"_final_img.jpg"), image)
    #cv2.waitKey(0)

    return image
def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    output_img = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    output_img[:r1, c:c + c1, :] = np.dstack([img2, img2, img2])

    # Go over all of the matching points and extract them
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw circles on the keypoints
        cv2.circle(output_img, (int(x1), int(y1)), 10, (0, 255, 255), 1)
        cv2.circle(output_img, (int(x2) + c, int(y2)), 10, (0, 255, 255), 1)

        # Connect the same keypoints
        cv2.line(output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 4)

    return output_img



def blending(kernel_size,prevkernel_size,image,image_mask,sec_image,sec_mask,weight):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if prevkernel_size !=0:
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (prevkernel_size, prevkernel_size))
    else:
        small_kernel=kernel
    prim_mask_erode = cv2.morphologyEx(image_mask, cv2.MORPH_ERODE, kernel, iterations=1)
    prim_mask_erode_small = cv2.morphologyEx(image_mask, cv2.MORPH_ERODE, small_kernel, iterations=1)
    prim_mask_padding = cv2.bitwise_and(cv2.bitwise_xor(image_mask, prim_mask_erode), sec_mask)
    prim_mask_padding_small = cv2.bitwise_and(cv2.bitwise_xor(image_mask, prim_mask_erode_small), sec_mask)
    if prevkernel_size != 0:
        prim_mask_padding = cv2.bitwise_xor(prim_mask_padding, prim_mask_padding_small)
    prim_mask_padding = prim_mask_padding[:, :, 0]
    prim_image_part = cv2.bitwise_and(image, image, mask=prim_mask_padding)
    sec_image_part = cv2.bitwise_and(sec_image, sec_image, mask=prim_mask_padding)
    padding = cv2.addWeighted(prim_image_part, weight, sec_image_part, 1-weight, 0.0)
    return padding

def trim(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = cv2.subtract(minRect, thresh)

    while cv2.countNonZero(sub) > 0:
        num_of_pixels=cv2.countNonZero(sub)
        sub_sums=[]

        subTop = sub[10:]
        sub_sums.append(np.sum(subTop))
        subBottom = sub[:-11]
        sub_sums.append(np.sum(subBottom))
        subLeft=sub[:,10:]
        sub_sums.append(np.sum(subLeft))
        subRight=sub[:,:-11]
        sub_sums.append(np.sum(subRight))

        min=np.argmin(sub_sums)
        if min==0:
            sub=sub[10:]
            image=image[10:]
        elif min==1:
            sub=sub[:-11]
            image=image[:-11]
        if min == 2:
            sub = sub[:,10:]
            image = image[:,10:]
        elif min==3:
            sub=sub[:,:-11]
            image=image[:,:-11]
        if num_of_pixels==cv2.countNonZero(sub):
            break
        #cv2.imshow("subs",sub)
        #cv2.waitKey(0)
    return image
