import imutils
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import base64
import io
import image_match

def alignment(imgs):
    debug = False

    transformed_imgs = [None] * len(imgs)
    template = imgs[0]
    transformed_imgs[0] = template
    # akaze = cv2.AKAZE_create()
    orb_detector = cv2.ORB_create(1000)
    kpts = []
    descs = []
    # kpts=[None]*len(imgs)
    # descs=[None]*len(imgs)

    kpt_template, desc_template = orb_detector.detectAndCompute(template, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    height, width, dim = np.shape(template)
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(1, len(imgs)):
        kpt, desc = orb_detector.detectAndCompute(imgs[i], None)

        matches = matcher.match(desc, desc_template, None)
        matches = sorted(matches, key=lambda x: x.distance)
        # keep only the top matches
        keep = int(len(matches) * 0.9)
        matches = matches[:keep]

        no_of_matches = len(matches)

        if debug:
            print(len(kpt))
            print(no_of_matches)
            image_kp = cv2.drawKeypoints(template, kpt_template, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            image_kp = imutils.resize(image_kp, width=1000)
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
        transformed_imgs[i] = (cv2.warpPerspective(imgs[i], homography, (width, height)))
    return transformed_imgs
def obj_remover(images):
    plt.style.use('grayscale')
    debug = False


    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    height, width, dim = np.shape(images[0])
    blankMask=np.ones((height,width))
    firstRun=True
    foregroundImages=[]
    masks=[]
    for frame in range(0,len(images)):
        print(frame)
        fgmask = fgbg.apply(images[frame])

        frame_open1 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel,iterations=1)
        frame_close = cv2.morphologyEx(frame_open1, cv2.MORPH_CLOSE, kernel,iterations=1)
       # frame_close = cv2.morphologyEx(frame_close, cv2.MORPH_CLOSE, kernel2,iterations=1)#cv2.dilate(fgmask,kernel)
        #frame_close = cv2.erode(frame_close, kernel2,iterations=3)
        frame_open = cv2.morphologyEx(frame_close, cv2.MORPH_OPEN, kernel2,iterations=1)#cv2.dilate(frame_open, kernel2)
        frame_open = fgmask.copy()#cv2.dilate(frame_close,kernel3,iterations=2)#fgmask.copy()#cv2.dilate(frame_open, kernel2)

        frame_open3= np.dstack([frame_open, frame_open, frame_open])

        contours, hierarchy = cv2.findContours(frame_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if frame >=1:
            hulls=[]
            max_area=0
            for cnt in range(0,len(contours)):
                hull = cv2.convexHull(contours[cnt])
                area1=cv2.contourArea(hull)
                area2 = cv2.contourArea(contours[cnt])
                if area1>0:
                    area_rat=area2/area1
                    if area_rat <=0.5:
                        #hulls[cnt] = contours[cnt]
                        hulls.append(contours[cnt])
                        max_area=area1
                        max_cnt=cnt
                        #print(max_area)
                    else:
                        hulls.append(hull)
                else:
                    hulls.append(hull)



            frame_open_view = imutils.resize(frame_open, width=1000)
            cv2.imshow('frame_open_pred', frame_open_view)

            cv2.fillPoly(frame_open, hulls, 255)


            frame_open_view = imutils.resize(frame_open, width=1000)
            cv2.imshow('frame_open', frame_open_view)
            cv2.waitKey(0)
        with_contours = cv2.drawContours(frame_open3, contours, -1, (255, 0, 255), 3)


        if firstRun:
            bitmask=cv2.bitwise_and(frame_open,frame_open)
            restHole=cv2.bitwise_xor(frame_open,frame_open)
            patchMask = cv2.bitwise_xor(frame_open,frame_open)
            firstRun = False
        else:
            globalMask = cv2.bitwise_xor(frame_open, frame_open)
            bitmask = cv2.bitwise_and(bitmask, frame_open)
            newMask = cv2.bitwise_not(frame_open)
            #cv2.imshow("fgmask",fgmask)
            #cv2.waitKey(0)


            foregroundImage = cv2.bitwise_and(images[frame], images[frame], mask=newMask)
            foregroundImages.append(foregroundImage)
            masks.append(newMask)
            if frame == 2:
                hole =(cv2.bitwise_and(frame_open, frame_open))
            if frame==(len(images)-1):
                #if frame==3:
                oldMask =  masks[frame-1]#cv2.bitwise_not(frame_open)
                coreImage = cv2.bitwise_and(images[frame], images[frame], mask=oldMask)
                coreImageGray = cv2.cvtColor(coreImage, cv2.COLOR_BGR2GRAY)
                ret, oldMask = cv2.threshold(coreImageGray,2, 255, cv2.THRESH_BINARY)

                coreImage = cv2.bitwise_and(images[frame], images[frame], mask=oldMask)
                cv2.imshow("coreimagepre",coreImage)
                #cv2.imshow("oldMask", oldMask)

                #elif frame<len(images):
                    #coreImage = cv2.bitwise_and(coreImage, coreImage, mask=oldMask)
                    #hole=(cv2.bitwise_and(hole,oldMask))
                    #invHole = cv2.bitwise_not(hole)
                #coreImage = cv2.bitwise_and(images[frame-1], images[frame-1], mask=oldMask)
                #newMask = cv2.bitwise_not(frame_open)
                #foregroundImage=cv2.bitwise_and(images[frame], images[frame], mask=newMask)
                invMask=cv2.bitwise_not(oldMask)
                #cv2.imshow("foreground", foregroundImage)
                #cv2.imshow("fgmask", fgmask)
                #cv2.imshow("frame_close", frame_close)
                #cv2.imshow("frame_open", frame_openn)
                #cv2.waitKey(0)
                patch = cv2.bitwise_or(foregroundImage, foregroundImage,mask=invMask)


                binPatch=cv2.bitwise_and(frame_open,invMask)
                binPatch=cv2.bitwise_xor(binPatch,invMask)
                #hole=invMask.copy()
                #cv2.imshow("hole",hole)
                hole=invMask.copy()#cv2.bitwise_or(invMask - binPatch,restHole)
                #hole = cv2.bitwise_or(invMask, bitmask)
                cv2.imshow("mask",hole)
                #cv2.imshow("invMask", invMask);
                oldMask = cv2.bitwise_not(frame_open)
                newMask3 = np.dstack([frame_open, frame_open, frame_open])
                #cv2.imshow("foreground+mask", foregroundImage+newMask3)
                #cv2.imshow("hole", hole)
                #cv2.show()
                temp=0
                avg_image = cv2.bitwise_or(foregroundImages[0],foregroundImages[0],mask=hole)
                #cv2.imshow("holepřed zmenšením", hole)
                for id in range(1,frame):
                    print(id)
                    holePatch=cv2.bitwise_or(foregroundImages[id],foregroundImages[id],mask=hole)
                    holePatchMask = cv2.bitwise_or(masks[id], masks[id], mask=hole)
                    #cv2.imshow("holePatchMaskpre", holePatchMask)
                    #if id < frame:
                    #    holePatchGray = cv2.cvtColor(holePatch, cv2.COLOR_BGR2GRAY)
                    #    ret, holePatchUpdate = cv2.threshold(holePatchGray, 2, 255, cv2.THRESH_BINARY)
                    #    holePatch = cv2.bitwise_or(foregroundImages[id], foregroundImages[id], mask=holePatchUpdate)

                    #else:
                    holePatchUpdate=holePatchMask#holePatch.copy()
                    #holePatchMask = cv2.bitwise_xor(holePatchMask, holePatchUpdate)
                    #cv2.imshow("holePatch", holePatch)
                    #cv2.imshow("holePatchUpdate", holePatchUpdate)
                    #cv2.waitKey(0)

                    hole = hole - holePatchUpdate
                    #hole = hole - holePatchMask
                    #hole=restHole.copy()
                    #holePatchMask3 = np.dstack([holePatchMask, holePatchMask, holePatchMask])
                    #cv2.imshow("hole",hole)
                    #cv2.imshow("holePatchMask",holePatchMask)
                    #cv2.imshow("hole", hole)
                    #cv2.imshow("holepatch", holePatch)
                    #cv2.waitKey(0)
                    patch=cv2.bitwise_or(holePatch,patch)
                    #patchMask=cv2.bitwise_or(holePatchUpdate,patchMask)



                #cv2.imshow("patchMask", patchMask)
                cv2.imshow("patch", patch)
                cv2.imshow("hole", hole)
                restHole=hole.copy()
                cv2.imshow("coreImage_pred", coreImage)

                #cv2.imshow("resthole", cv2.bitwise_xor(patchMask,patchUpdate))

                #cv2.imshow("updatedpatch", patch)

                #cv2.imshow("resthole", restHole)
                #cv2.imshow("bitmask", bitmask)
                #cv2.waitKey(0)
                #holePatch = cv2.bitwise_or(foregroundImages[maxId], foregroundImages[maxId], mask=hole)

                #hsv = cv2.cvtColor(holePatch, cv2.COLOR_BGR2HSV)
                #black = np.array([0, 0, 0], np.uint8)
                #restHole = hole-cv2.inRange(hsv, black, black)
                #cv2.imshow("restHole", restHole)
                #cv2.waitKey(0)
                #cv2.imshow("hole",hole)
                #cv2.imshow("holePatch",holePatch)
                #cv2.imshow("holePatchMask",holePatchMask)
                #cv2.imshow("restHole", restHole)
                #cv2.waitKey(0)
                #completePatch=cv2.bitwise_or(holePatch,patch)

                invMask3 = np.dstack([invMask, invMask, invMask])
                patchMask3=np.dstack([patchMask, patchMask, patchMask])
                bitmask3 = np.dstack([bitmask, bitmask, bitmask])

                #cv2.bitwise_not(patch,mask=patchMask)

                #patchMaskhole=  patch.patchMask3
                #cv2.imshow("patchMaskhole-patch", patchMaskhole-patch)
                #cv2.imshow("patchMaskhole", patchMaskhole)
                #cv2.imshow("patchMask", patchMask)
                #cv2.imshow("patch+ patchMask", patch+patchMask3)
                #cv2.imshow("patch", patch)
                #cv2.imshow("PredcoreImage", coreImage)
                #cv2.imshow("coreimage+patchMask", coreImage+patchMask3)
                #cv2.imshow("coreimage+patchMask-coreimahe", coreImage + patchMask3-coreImage)
                #cv2.imshow("frame_open",frame_open)
                #cv2.waitKey(0)
                coreImage = cv2.bitwise_or(coreImage, patch)
                cv2.imshow("hole_final", hole)



                repairedCoreImage = cv2.inpaint(coreImage, hole, 3, cv2.INPAINT_TELEA
                                                )

                cv2.imshow("repairedCoreImage",repairedCoreImage)
                #patchGray = cv2.cvtColor(coreImage, cv2.COLOR_BGR2GRAY)
                #ret, coreImageMask = cv2.threshold(patchGray, 0, 255, cv2.THRESH_BINARY)
                #restHole=cv2.bitwise_or(restHole,cv2.bitwise_not(coreImageMask))
                pixelHole = coreImage+invMask3
                #cv2.imshow("invMask", invMask)
                #cv2.imshow("pixelHole", pixelHole)
                #cv2.imshow("holePatchGrayinv", cv2.bitwise_not(holePatchGray))
                #cv2.imshow("coreImage", coreImage)
                #cv2.imshow("restHole", restHole)
                #cv2.imshow("bitmask", bitmask)
                #cv2.waitKey(0)

                #hole =(cv2.bitwise_or(hole,(binPatch)))
                #invHole=cv2.bitwise_and(invHole,cv2.bitwise_not(hole))
                #cv2.imshow("patch", patch)
                #cv2.imshow("holePatch", holePatch)
                # cv2.imshow("hole", hole)

                #cv2.imshow("precoreimage", coreImage)
                #coreImage = cv2.bitwise_or(coreImage, holePatch)
                #cv2.imshow("coreImageMask", coreImageMask)
                cv2.imshow("coreimage",coreImage)
                cv2.waitKey(0)
                if debug:
                    cv2.imshow("foreground", foregroundImage)
                    cv2.waitKey(0)
                    cv2.imshow("patch", patch)
                    cv2.waitKey(0)
                    plt.imshow(hole)
                    plt.show()
                    plt.imshow(patch)
                    plt.show()
                    cv2.imshow("coreImage", coreImage)
                    cv2.waitKey(0)

    #
    #backgound_image=fgbg.getBackgroundImage()

    cv2.imshow("coreImage",coreImage )
    cv2.waitKey(0)
    #plt.imshow(bitmask)
    #plt.show()

    return bitmask