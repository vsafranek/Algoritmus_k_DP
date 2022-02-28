import numpy as np
import cv2
from PIL import Image
import base64
import io

def main(data):

    #datas=unsplitData.split("#")
    #imgs=[]
    #for data in datas:
    decoded_data= base64.b64decode(data)
    np_data=np.fromstring(decoded_data,np.uint8)
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    #imgs.append(img)

    #for img in imgs:
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)





    pil_im =Image.fromarray(img_gray)
    buff = io.BytesIO()
    pil_im.save(buff,format="JPEG")
    img_str= base64.b64encode(buff.getvalue())

    return ""+str(img_str,'utf-8')