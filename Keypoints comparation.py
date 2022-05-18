import cv2
import numpy
import time

park = cv2.imread("CompImages/Dataset_09/IMG_3085.jpeg")
plakat = cv2.imread("CompImages/Dataset_07/IMG_3094.jpeg")
katedrala = cv2.imread("CompImages/Dataset_05/IMG_3036.jpeg")

template = katedrala
detector_type="akaze"

start_time = time.time()
if detector_type == "akaze":
    detector = cv2.AKAZE_create()
elif detector_type == "brisk":
    detector = cv2.BRISK_create()
elif detector_type == "orb":
    detector = cv2.ORB_create(1000)
elif detector_type == "sift":
    keypoint_limit = 3000
    detector = cv2.SIFT_create()
else:
    detector = cv2.ORB_create(10000)

keypoints, descriptors = detector.detectAndCompute(template, None)

end_time=(time.time() - start_time)
name=str("katedrala"+"_"+detector_type+".jpg")
print(name)
print(len(keypoints))
print(template.shape)
print((end_time / len(keypoints))*1000)
print("--- %s seconds ---" % end_time)

image=cv2.drawKeypoints(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), keypoints,None, (255, 0, 255))
image = image.astype('uint8')

cv2.imwrite(name,image)
cv2.imshow("katedrala",  image)
cv2.waitKey(0)