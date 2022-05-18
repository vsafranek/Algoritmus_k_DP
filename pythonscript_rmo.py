import time

import numpy as np
import cv2
from PIL import Image
import base64
import io
import glob
import os
import RemoveObjects_process as ro

def remove_objects(countImageString):
    imgs = []
    countImage=int(countImageString)

    # datas = image_string.split("#")
    # for data in datas:
    #     decoded_data = base64.b64decode(data)
    #     np_data = np.fromstring(decoded_data, np.uint8)
    #     img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    #     imgs.append(img)
    # print("pocet obrazku", len(imgs))

    path='/storage/emulated/0/SmartCamera/'
    valid_images=[".jpg",".png",".jpeg"]
    #for filename in sorted(glob.glob(path + '\*jpeg')):
    id=0
    while id < countImage:
        if (id<len(os.listdir(path))):
            f =sorted(os.listdir(path))[id]
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
                 #imgs.append(Image.open(os.path.join(path,f)))
            im =cv2.imread(os.path.join(path,f))
            imgs.append(im)
            id=id+1
        else:
            time.sleep(0.5)


    trans_imgs = ro.alignment(imgs)
    for i in range(0, len(trans_imgs)):
        print(np.shape(trans_imgs[i]))
    final_img = ro.obj_remover(trans_imgs)

    pil_im = Image.fromarray(final_img)
    buff = io.BytesIO()
    pil_im.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())

    return ""+str(img_str,'utf-8')

