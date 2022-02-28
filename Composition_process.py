import cv2
import imutils
import numpy as np

def panorama_orb(imgs,detector_type):
    global prim_image_mask_old
    debug = False

    transformed_imgs = [None] * len(imgs)
    template = imgs[0]
    transformed_imgs[0] = template
    if detector_type == "akaze":
        detector = cv2.AKAZE_create()
    elif detector_type == "brisk":
        detector = cv2.BRISK_create()
    elif detector_type == "orb":
        detector = cv2.ORB_create(3000)
    elif detector_type == "sift":
        detector = cv2.SIFT_create(3000)
    kpts = []
    descs = []
    # kpts=[None]*len(imgs)
    # descs=[None]*len(imgs)
    # Create our ORB detector and detect keypoints and descriptors

    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    img = imgs[0]
    for i in range(1, len(imgs)):

        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = detector.detectAndCompute(template, None)
        keypoints2, descriptors2 = detector.detectAndCompute(imgs[i], None)
        print("pocet nalezených keypointů: ",len(keypoints1),len(keypoints2))
        # It will find all of the matching keypoints on two images
        if detector_type in ["sift"]:
            matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            matches = matcher.match(descriptors1, descriptors2, None)
            matches = sorted(matches, key=lambda x: x.distance)
            keep = int(len(matches) * 0.8)
            best_matches = matches[:keep]


        elif detector_type in ["orb", "akaze", "brisk"]:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(descriptors1, descriptors2, None)
            matches = sorted(matches, key=lambda x: x.distance)
            keep = int(len(matches) * 0.8)
            best_matches = matches[:keep]
        #best_matches = []
        #for m, n in matches:
        #    if m.distance < 0.8 * n.distance:
        #       best_matches.append(m)

        no_of_matches = len(best_matches)
        print("best matches",no_of_matches)
        # Set minimum match condition
        MIN_MATCH_COUNT = 10
        w = 9072#template.shape[1] + imgs[i].shape[1]
        h = 8064#template.shape[0]
        if debug:
            cv2.imshow("matches",draw_matches(cv2.cvtColor(template,cv2.COLOR_BGR2GRAY), keypoints1, cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY), keypoints2, best_matches[:30]))
            cv2.imshow("template",cv2.drawKeypoints(template, [keypoints1[m.queryIdx] for m in best_matches], None, (255, 0, 255)))
            cv2.imshow("imgs[s]",cv2.drawKeypoints(imgs[i], [keypoints2[m.trainIdx] for m in best_matches], None, (255, 0, 255)))
            cv2.waitKey(0)
        if len(best_matches) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography
            src_pts = np.float32([keypoints1[m.queryIdx].pt for (j ,m) in enumerate(best_matches)]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for (j, m) in enumerate(best_matches)]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            #dst=warpImages(template,imgs[i],homography)
            dst = cv2.warpPerspective(imgs[i], homography, (w, h))

            #cv2.imshow("dst",dst)
            #cv2.waitKey(0)

            SecImage_Mask = np.zeros(imgs[i].shape, dtype=np.uint8)
            SecImage_Mask[:, :, :] = 255

            dst_Mask = cv2.warpPerspective(SecImage_Mask, homography, (w, h))
            try:
                prim_image_mask = np.zeros(dst_Mask.shape, dtype=np.uint8)
                prim_image_mask[0:prim_image_mask_old.shape[0], 0:prim_image_mask_old.shape[1], :] = prim_image_mask_old

                prim_image_mask_old=cv2.bitwise_or(dst_Mask,prim_image_mask)
            except:
                prim_image_mask = np.zeros(dst_Mask.shape, dtype=np.uint8)
                prim_image_mask[0:template.shape[0], 0:template.shape[1], :] = 255

                prim_image_mask_old=cv2.bitwise_or(dst_Mask,prim_image_mask)

            prim_image = np.zeros(dst_Mask.shape, dtype=np.uint8)
            prim_image[0:template.shape[0], 0:template.shape[1], :] = template

            sec_mask = cv2.bitwise_and(cv2.bitwise_or(dst_Mask, prim_image_mask),cv2.bitwise_not(prim_image_mask))
            Plus_Image = cv2.bitwise_and(dst, dst, mask=sec_mask[:, :, 0])
            Image=cv2.bitwise_or(Plus_Image,prim_image)
            if debug:
                cv2.imshow("secondary image",dst)
                cv2.imshow("secondary image mask", dst_Mask)
                cv2.imshow("primary image mask", prim_image_mask)
                cv2.imshow(" mask", sec_mask)
                cv2.imshow("image",Image)
                cv2.imshow("plus image", Plus_Image)
                cv2.waitKey(0)
            template = np.copy(Image)
        #
        #dst[0:template.shape[0], 0:template.shape[1]] = template

        #dst = warpImages(template, imgs[i], homography)
        #w = dst.shape[1]
        #h = dst.shape[0]

        #template = np.copy(trim_side(dst, h, w))
        #w = template.shape[1]
        #h = template.shape[0]
        cv2.imshow("final pic",template)
        cv2.waitKey(0)
    return dst
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