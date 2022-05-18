
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import cv2
from sewar.full_ref import mse,ssim, rmse

def image_detector(imgs, detector_type):
    debug = True

    transformed_imgs = [None] * len(imgs)
    template = imgs[0]

    transformed_imgs[0] = template
    if detector_type == "akaze":
        detector = cv2.AKAZE_create()
    elif detector_type == "brisk":
        detector = cv2.BRISK_create()
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
        keep = int(len(matches) * 0.90)
        good = matches[:keep]


    elif detector_type in ["orb", "akaze", "brisk"]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.match(desc, desc_template, None)
        matches = sorted(matches, key=lambda x: x.distance)
        keep = int(len(matches) * 0.9)
        good = matches[:keep]

    no_of_matches = len(good)

    if debug:
        #cv2.imshow("img1",template)
        #cv2.imshow("img2",imgs[1])

        print("pocet kpt obr1: ", len(kpt_template))
        print("pocet kpt obr2: ", len(kpt))

        print("Pocet prirazenych keypointu: ", no_of_matches)
        #cv2.waitKey(0)
    return [len(kpt_template),len(kpt),no_of_matches,good]
def blurred_filter(kpt_data,kpt_match_data):
    kpt_max=max(kpt_data)
    top5=sorted(kpt_data)[len(kpt_data)-5:]
    kpt_mean_max=sum(top5)/5
    krit=80
    print("kpt_mean_max: ",kpt_mean_max)
    kpt_match_max= max(kpt_match_data)
    print("maximální počer keypointů: ",kpt_max)
    filter=[]
    for i in range(0,len(kpt_data)):
        if kpt_data[i]<kpt_mean_max/100*krit:
            filter.append(0)
        else:
            filter.append(1)
    return filter
def repeating_filter(imgs,matches_data):
    krit=10
    filter=[1]
    mse_krit=[]
    mean_dist=[]
    dist_mean_max=[]
    for i in range(1,len(imgs)):
        dist = 0
        dist = [m.distance for m in matches_data[i-1]]
        top10 = sorted(dist)[len(dist) - int(np.floor(len(dist)*0.1)):]
        dist_mean_max.append( sum(top10) / int(np.floor(len(dist)*0.1)))
        #print(dist_mean_max)
        mean_dist.append(sum(dist) / len(dist))
        #median_dist = st.mode(dist)
        mse_krit.append(mse(imgs[i-1],imgs[i]))
        print("mean_dist ", mean_dist[i-1])
        #print("median_dist pro ", median_dist)
        print("mse ", mse_krit[i-1])
        krit = 70
        print("krit: ",i-1," ",dist_mean_max[i-1]/100*krit)
        if np.floor(mean_dist[i-1])<=dist_mean_max[i-1]/100*krit:
            #cv2.imshow(("img1 "+str(i-1)),imgs[i-1])
            cv2.imshow(("remove "+str(i)), imgs[i])
            #cv2.waitKey(0)
            filter.append(0)
        else:
            #cv2.imshow(("img1 " + str(i - 1)), imgs[i - 1])
            cv2.imshow(("nechani "+str(i)), imgs[i])
            filter.append(1)
        #cv2.waitKey(0)
    plt.plot(range(1,len(imgs)),mse_krit)
    plt.show()
    plt.plot(range(1, len(imgs)), mean_dist)
    plt.show()
    print(filter)
    return filter