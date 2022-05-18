import numpy as np
import cv2
from PIL import Image
import base64
import io
import glob
import os
import RemoveObjects_process as ro
import ReturnTime_process as rt
def main():
    imgs=[]
    path='/storage/emulated/0/SmartCamera/'
    valid_images=[".jpg",".png",".jpeg"]
    #for filename in sorted(glob.glob(path + '\*jpeg')):
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
             #imgs.append(Image.open(os.path.join(path,f)))
        im =cv2.imread(os.path.join(path,f))
        imgs.append(im)
    #datas=unsplitData.split("#")
    #imgs=[]
    #for data in datas:
    #decoded_data= base64.b64decode(data)
    #np_data=np.fromstring(decoded_data,np.uint8)
    #img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    #imgs.append(img)
    print("pocet obrazku nalezenych ",len(imgs))
    for img in imgs:
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)





    pil_im =Image.fromarray(img_gray)
    buff = io.BytesIO()
    pil_im.save(buff,format="JPEG")
    img_str= base64.b64encode(buff.getvalue())

    return ""+str(img_str,'utf-8')

def super_composition(data):
    decoded_data= base64.b64decode(data)
    np_data=np.fromstring(decoded_data,np.uint8)
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    #imgs.append(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    pil_im =Image.fromarray(img_gray)
    buff = io.BytesIO()
    pil_im.save(buff,format="JPEG")
    img_str= base64.b64encode(buff.getvalue())

    return ""+str(img_str,'utf-8')

def remove_objects(image_string):
    imgs = []

    datas = image_string.split("#")
    for data in datas:
        decoded_data = base64.b64decode(data)
        np_data = np.fromstring(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        imgs.append(img)
    print("pocet obrazku", len(imgs))

    #     path='/storage/emulated/0/SmartCamera/'
    #     valid_images=[".jpg",".png",".jpeg"]
    #     #for filename in sorted(glob.glob(path + '\*jpeg')):
    #     for f in os.listdir(path):
    #         ext = os.path.splitext(f)[1]
    #         if ext.lower() not in valid_images:
    #             continue
    #              #imgs.append(Image.open(os.path.join(path,f)))
    #         im =cv2.imread(os.path.join(path,f))
    #         imgs.append(im)

    trans_imgs = ro.alignment(imgs)
    for i in range(0, len(trans_imgs)):
        print(np.shape(trans_imgs[i]))
    final_img = ro.obj_remover(trans_imgs)

    pil_im = Image.fromarray(final_img)
    buff = io.BytesIO()
    pil_im.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())

    return ""+str(img_str,'utf-8')

def timeback(image_string):
    imgs = []

    datas = image_string.split("#")
    for data in datas:
        decoded_data = base64.b64decode(data)
        np_data = np.fromstring(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        imgs.append(img)
    rfilter=bool(imgs[-1])
    imgs.remove(-1)
    bfilter=bool(imgs[-1])
    imgs.remove(-1)

    i = 0
    kpt_data = []
    kpt_match_data = []
    while i < len(imgs) - 1:
        print("obrazek: ", i + 1, " a ", i + 2)
        scale_percent = 5
        width = 400  # int(imgs[i].shape[1] * scale_percent / 100)
        height = int(imgs[i].shape[0] * width / imgs[i].shape[1])
        dim = (width, height)

        img_small1 = cv2.resize(imgs[i], dim, interpolation=cv2.INTER_CUBIC)
        img_small2 = cv2.resize(imgs[i + 1], dim, interpolation=cv2.INTER_CUBIC)
        choosed_imgs = [img_small1, img_small2]
        [kpt1, kpt2, kptM] = rt.ReturnTime_process.image_detector(choosed_imgs, "brisk")
        kpt_data.append(kpt1)
        kpt_match_data.append(kptM)
        i = i + 1
    if bfilter == True:
        bfilter = rt.ReturnTime_process.blurred_filter(kpt_data, kpt_match_data)
    else:
        bfilter = np.ones(len(imgs), dtype=int)
    if rfilter == True:
        rfilter = rt.ReturnTime_process.repeating_filter(kpt_data, kpt_match_data)
    else:
        rfilter = np.ones(len(imgs), dtype=int)
    filter = []
    for i in range(0, len(bfilter)):
        if bfilter[i] == 1 and rfilter[i] == 1:
            filter.append(1)
            print(kpt_data[i])
        else:
            filter.append(0)

    return ("".join(filter))