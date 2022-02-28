
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt

def image_detector(imgs, detector_type):
    debug = True

    transformed_imgs = [None] * len(imgs)
    template = imgs[0]

    transformed_imgs[0] = template
    if detector_type == "akaze":
        detector = cv2.AKAZE_create()
    elif detector_type == "brisk":
        detector = cv2.BRISK_create(30,1)
    elif detector_type == "orb":
        detector = cv2.ORB_create(7000)
    elif detector_type == "sift":
        detector = cv2.SIFT_create(3000)

    kpt_template, desc_template = detector.detectAndCompute(template, None)
    kpt, desc = detector.detectAndCompute(imgs[1], None)

    if detector_type in ["sift"]:
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(desc, desc_template, None)
        matches = sorted(matches, key=lambda x: x.distance)
        keep = int(len(matches) * 0.8)
        good = matches[:keep]


    elif detector_type in ["orb", "akaze", "brisk"]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc, desc_template, None)
        matches = sorted(matches, key=lambda x: x.distance)
        keep = int(len(matches) * 0.8)
        good = matches[:keep]

    no_of_matches = len(good)

    if debug:

        print("pocet kpt obr1: ", len(kpt_template))
        print("pocet kpt obr2 : ", len(kpt))

        print("Pocet prirazenych keypointu: ", no_of_matches)
        image_kp = cv2.drawKeypoints(imgs[1], kpt, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        image_kp = imutils.resize(image_kp, width=1000)
        #cv2.imshow("Draw Keypoints", image_kp)

        main_image_kp = cv2.drawKeypoints(template, kpt_template, None,
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        main_image_kp = imutils.resize(main_image_kp, width=1000)
        #cv2.imshow(" Keypoints", main_image_kp)
        #cv2.waitKey(0)

def mser_detector(img):
    mser=cv2.MSER_create()
    gray = cv2.cvtColor(img, 'RGB2GRAY');
    [regions, rects] = mser.detectRegions(gray)