import imutils
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import base64
import io


def alignment(imgs):
    debug = False

    transformed_imgs = [None] * len(imgs)
    template = imgs[0]#[len(imgs)-2]
    print(template.shape)
    transformed_imgs[0]=template#[len(imgs)-2] = template
    # akaze = cv2.ORB_create()
    orb_detector = cv2.AKAZE_create()
    kpts = []
    descs = []
    # kpts=[None]*len(imgs)
    # descs=[None]*len(imgs)

    kpt_template, desc_template = orb_detector.detectAndCompute(template, None)
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    height, width, dim = np.shape(template)
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #image_kp = cv2.drawKeypoints(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), kpt_template, None, (255, 0, 255))
    #cv2.imwrite(("RO_data/keyipoint_image_" + str(0) + ".jpg"), image_kp)
    for i in range(1, len(imgs)):
        kpt, desc = orb_detector.detectAndCompute(imgs[i], None)
        print(i)


        matches = matcher.match(desc, desc_template, None)
        matches = sorted(matches, key=lambda x: x.distance)
        # keep only the top matches
        keep = int(len(matches) * 0.9)
        matches = matches[:keep]

        no_of_matches = len(matches)

        if debug:
            print(len(kpt))
            print(no_of_matches)
            #image_kp = cv2.drawKeypoints(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY), kpt, None, (255, 0, 255))
            image_kp = cv2.drawKeypoints(template, kpt_template, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #image_debug = cv2.cvtColor(image_kp, cv2.COLOR_GRAY2RGB)

            #cv2.imwrite(("RO_data/keyipoint_image_" + str(i) + ".jpg"), image_kp)

            image_kp = imutils.resize(image_kp, width=800)
            cv2.imshow("Matched Keypoints", image_kp)
            cv2.waitKey(0)

            matchedVis = cv2.drawMatches(imgs[i], kpt, template, kpt_template,
                                         matches, None)
            matchedVis = imutils.resize(matchedVis, width=2000)
            cv2.imshow("Matched Keypoints", matchedVis)
            cv2.waitKey(0)

        p1 = np.zeros((no_of_matches, 2), dtype="float")
        p2 = np.zeros((no_of_matches, 2), dtype="float")

        for (j, m) in enumerate(matches):
            p1[j] = kpt[m.queryIdx].pt
            p2[j] = kpt_template[m.trainIdx].pt

        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        if i ==len(imgs)-2 :
            #pass
            transformed_imgs[i] = (cv2.warpPerspective(imgs[i], homography, (width, height)))
        else:
            transformed_imgs[i] = (cv2.warpPerspective(imgs[i], homography, (width, height)))
    for item in transformed_imgs:
        pass
        #cv2.imshow("transformed_img", item)
        #cv2.waitKey(0)
    return transformed_imgs


def obj_remover(images):
    plt.style.use('grayscale')
    debug = False
    debug1 = False
    debug2 = False
    debug3 = False
    debug4 = False
    debug5 = False


    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    # fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel30 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    height, width, dim = np.shape(images[0])
    blankMask = np.ones((height, width))
    first_run = True
    foregroundImages = []
    masks = []
    scale_percent = 12.5  # percent of original size
    width=images[0].shape[1]
    height=images[0].shape[0]
    width_small = 400
    height_small = int(height *width_small/width)
    dim = (width, height)
    dim_small = (width_small,height_small)
    print(dim_small)
    for frame in range(0, len(images)):
        print(frame)


        small_image=cv2.resize(images[frame], dim_small, interpolation=cv2.INTER_AREA)

        fgmask = fgbg.apply(small_image)
        if debug1 == True:

            # image = [fgmask_debug,fgmask_debug,fgmask_debug]
            # image = image.astype('uint8')
            image_debug = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            #
            #cv2.imwrite(("raw_image_" + str(frame) + ".jpg"), image_debug)

        #frame_open_view = imutils.resize(fgmask, width=1000)
        #cv2.imshow('fgmask', frame_open_view)
        #frame_open_view = imutils.resize(fgmask_small, width=1000)
        #cv2.imshow('fgmask_small', frame_open_view)

        morphology = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        morphology = cv2.morphologyEx(morphology, cv2.MORPH_CLOSE, kernel3, iterations=1)
        # frame_close = cv2.morphologyEx(frame_close, cv2.MORPH_CLOSE, kernel2,iterations=1)#cv2.dilate(fgmask,kernel)
        # frame_close = cv2.erode(frame_close, kernel2,iterations=3)
        # morphology = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel2,iterations=1)#cv2.dilate(frame_open, kernel2)
        morphology = cv2.dilate(morphology, kernel3, iterations=1)  # fgmask.copy()#cv2.dilate(frame_open, kernel2)

        frame_open = np.copy(morphology)

        if debug2 == True:
            debug_image=np.dstack([frame_open,frame_open,frame_open])
            cv2.imwrite(("RO_data/dilate_mask_" + str(frame) + ".jpg"), debug_image)

        #cv2.imshow("frame open",frame_open)
        #cv2.waitKey(0)
        if frame >= 1:
            frame_open_view = imutils.resize(frame_open, width=1000)
            #cv2.imshow('frame_open_pred', frame_open_view)

            frame_open = make_contures(frame_open)
            frame_open = cv2.morphologyEx(frame_open, cv2.MORPH_CLOSE, kernel7, iterations=1)
            frame_open = make_contures(frame_open)

            frame_open_view = imutils.resize(images[frame], width=1000)
            #cv2.imshow('original image', frame_open_view)

            frame_open_view = imutils.resize(frame_open, width=1000)
            #cv2.imshow('frame_open_po', frame_open_view)

            #cv2.waitKey(0)
        # with_contours = cv2.drawContours(frame_open3, contours, -1, (255, 0, 255), 3)

        frame_open = cv2.resize(frame_open, dim, interpolation=cv2.INTER_AREA)
        if debug3 == True:
            debug_image=np.dstack([frame_open,frame_open,frame_open])
            cv2.imwrite(("RO_data/full_mask_" + str(frame) + ".jpg"), debug_image)

        if first_run:
            bitmask = cv2.bitwise_and(frame_open, frame_open)
            restHole = cv2.bitwise_xor(frame_open, frame_open)
            patchMask = cv2.bitwise_xor(frame_open, frame_open)
            first_run = False
        else:
            globalMask = cv2.bitwise_xor(frame_open, frame_open)
            bitmask = cv2.bitwise_and(bitmask, frame_open)
            newMask = cv2.bitwise_not(frame_open)
            # cv2.imshow("fgmask",fgmask)
            # cv2.waitKey(0)

            foregroundImage = cv2.bitwise_and(images[frame], images[frame], mask=newMask)
            foregroundImages.append(foregroundImage)

            if debug4 == True:
                cv2.imwrite(("RO_data/hole_image_" + str(frame) + ".jpg"), foregroundImage)
            masks.append(newMask)
            if frame == 2:
                hole = (cv2.bitwise_and(frame_open, frame_open))
            if frame == (len(images) - 1):
                # if frame==3:
                oldMask = masks[frame-1]  # cv2.bitwise_not(frame_open)
                core_image = cv2.bitwise_and(images[0], images[0], mask=oldMask)
                coreImageGray = cv2.cvtColor(core_image, cv2.COLOR_BGR2GRAY)
                ret, oldMask = cv2.threshold(coreImageGray, 2, 255, cv2.THRESH_BINARY)

                core_image = cv2.bitwise_and(images[frame], images[frame], mask=oldMask)
                #cv2.imshow("core_image",core_image)
                #cv2.imshow("image",images[frame])
                #cv2.waitKey(0)
                invMask = cv2.bitwise_not(oldMask)


                binPatch = cv2.bitwise_and(frame_open, invMask)
                binPatch = cv2.bitwise_xor(binPatch, invMask)
                # hole=invMask.copy()
                # cv2.imshow("hole",hole)
                hole = invMask.copy()  # cv2.bitwise_or(invMask - binPatch,restHole)

                padding_num = 4
                kernel30 = cv2.getStructuringElement(cv2.MORPH_RECT, (padding_num * 20, padding_num * 20))
                hole_dilate = cv2.morphologyEx(hole, cv2.MORPH_DILATE, kernel30, iterations=1)
                patch = cv2.bitwise_or(foregroundImage, foregroundImage, mask=invMask)
                patchDilate = patch.copy()
                for id in range(1,frame):
                    print(id)

                    holePatch = cv2.bitwise_or(foregroundImages[id], foregroundImages[id], mask=hole)
                    holePatchMask = cv2.bitwise_or(masks[id], masks[id], mask=hole)

                    holePatchDilate = cv2.bitwise_or(foregroundImages[id], foregroundImages[id], mask=hole_dilate)
                    holePatchMaskDilate = cv2.bitwise_or(masks[id], masks[id], mask=hole_dilate)


                    holePatchUpdate = holePatchMask

                    hole = hole - holePatchUpdate
                    hole_dilate = hole_dilate - holePatchMaskDilate

                    patchDilate= cv2.bitwise_or(holePatchDilate, patchDilate)
                    patch = cv2.bitwise_or(holePatch, patch)

                    patchMaskDilate=cv2.bitwise_or(holePatchMaskDilate,patchMask)
                    patchMask=cv2.bitwise_or(holePatchMask,patchMask)

                core_image_mask= cv2.bitwise_not(patchMask)

                padding = blending(20, 0, core_image, core_image_mask, patchDilate, patchMaskDilate, 0.20)
                padding += blending(40, 20, core_image, core_image_mask, patchDilate, patchMaskDilate, 0.40)
                padding += blending(60, 40, core_image, core_image_mask, patchDilate, patchMaskDilate, 0.60)
                padding += blending(80, 60, core_image, core_image_mask, patchDilate, patchMaskDilate, 0.80)


                kernel30 = cv2.getStructuringElement(cv2.MORPH_RECT, (padding_num * 20, padding_num * 20))
                prim_mask_erode = cv2.morphologyEx(core_image_mask, cv2.MORPH_ERODE, kernel30, iterations=1)
                share_mask = cv2.bitwise_xor(core_image_mask, prim_mask_erode)

                core_image = cv2.bitwise_or(core_image, patch)
                image_holed = cv2.bitwise_or(core_image, core_image, mask=cv2.bitwise_not(share_mask))

                new_image = image_holed + padding

                #cv2.imshow("patchMask", patchMask)
                #cv2.imshow("patch", patch)
                if debug5 == True:
                    #cv2.imwrite(("RO_data/patch.jpg"), patch)
                    cv2.imwrite(("RO_data/final.jpg"), trim(crop(new_image)))

                    cv2.imshow("patchDilate",patchDilate)
                    cv2.imshow("core_image_mask", core_image_mask)
                    cv2.imshow("image_holed",image_holed)
                    cv2.imshow("coreImage", core_image)
                    cv2.imshow("padding",padding)
                    #cv2.imshow("new_image",trim(crop(new_image)))
                    cv2.waitKey(0)



                #smoothed_edge = cv2.addWeighted(coreImage, 0.5, hole_padding, 0.5, 0.0)
                #cv2.imshow("hole_final", hole)
                #cv2.imshow("smoothed_edge", smoothed_edge)
                repaired_core_image = cv2.inpaint(new_image, hole, 3, cv2.INPAINT_TELEA)

                #cv2.imshow("repaired_core_image", repaired_core_image)

                #cv2.waitKey(0)
                if debug:
                    cv2.imshow("foreground", foregroundImage)
                    cv2.waitKey(0)
                    cv2.imshow("patch", patch)
                    cv2.waitKey(0)
                    cv2.imshow("coreImage", core_image)
                    cv2.waitKey(0)

    #cv2.imshow("coreImage", core_image)
    #cv2.waitKey(0)
    # plt.imshow(bitmask)
    # plt.show()

    return repaired_core_image #trim(crop(repaired_core_image))#


def make_contures(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = []
    max_area = 0
    print("pocet countours", len(contours))
    for cnt in range(0, len(contours)):
        hull = cv2.convexHull(contours[cnt])
        area1 = cv2.contourArea(hull)
        area2 = cv2.contourArea(contours[cnt])
        if area1 > 0:
            area_rat = area2 / area1
            if area_rat <= 0.5:
                # hulls[cnt] = contours[cnt]
                hulls.append(contours[cnt])
                max_area = area1
                max_cnt = cnt
                # print(max_area)
            else:
                hulls.append(hull)
        else:
            hulls.append(hull)
    cv2.fillPoly(image, hulls, 255)
    return image

def blending(kernel_size,prevkernel_size,image,image_mask,sec_image,sec_mask,weight):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if prevkernel_size !=0:
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (prevkernel_size, prevkernel_size))
    else:
        small_kernel = kernel
    prim_mask_erode = cv2.morphologyEx(image_mask, cv2.MORPH_ERODE, kernel, iterations=1)
    prim_mask_erode_small = cv2.morphologyEx(image_mask, cv2.MORPH_ERODE, small_kernel, iterations=1)
    prim_mask_padding = (cv2.bitwise_xor(image_mask, prim_mask_erode))
    prim_mask_padding_small = cv2.bitwise_xor(image_mask, prim_mask_erode_small)
    if prevkernel_size != 0:
        prim_mask_padding = cv2.bitwise_xor(prim_mask_padding, prim_mask_padding_small)

    prim_image_part = cv2.bitwise_and(image, image, mask=prim_mask_padding)
    sec_image_part = cv2.bitwise_and(sec_image, sec_image, mask=prim_mask_padding)

    padding = cv2.addWeighted(prim_image_part, weight, sec_image_part, 1-weight, 0.0)
    return padding

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


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

def optic_flow(imgs):
    prvs = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(imgs[0])
    hsv[..., 1] = 255
    id=0

    for image in imgs[:]:
        next = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #cv2.imshow('frame2', bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            pass
            #cv2.imwrite('opticalfb.png', image)
            #cv2.imwrite('opticalhsv.png', bgr)

        #mask=cv2.threshold(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 4, 254, cv2.THRESH_BINARY)[1]
        #cv2.imshow("bgr", mask)
        #cv2.imwrite(("RO_data/optical"+str(id)+".jpg"),mask)
        #cv2.waitKey(0)
        prvs = next
        id+=1
