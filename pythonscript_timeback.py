import numpy as np
import cv2
import base64
import ReturnTime_process as rt


def timeback(image_string):
    imgs = []

    datas = image_string.split("#")

    rfilter=bool(datas[-1])
    datas.pop()
    bfilter=bool(datas[-1])
    datas.pop()

    print("b filter status",bfilter)
    print("r filter status",rfilter)

    for data in datas:
        decoded_data = base64.b64decode(data)
        np_data = np.fromstring(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        imgs.append(img)
    print("pocet nactenych obrazku",len(imgs))


    i = 0
    kpt_data = []
    kpt_match_data = []
    while i < len(imgs) - 1:
        print("obrazek: ", i + 1, " a ", i + 2)
        scale_percent = 5
        print(imgs[i].shape)
        width = 400  # int(imgs[i].shape[1] * scale_percent / 100)
        height = int(imgs[i].shape[0] * width / imgs[i].shape[1])
        dim = (width, height)

        img_small1 = cv2.resize(imgs[i], dim, interpolation=cv2.INTER_CUBIC)
        img_small2 = cv2.resize(imgs[i + 1], dim, interpolation=cv2.INTER_CUBIC)
        choosed_imgs = [img_small1, img_small2]
        [kpt1, kpt2, kptM] = rt.image_detector(choosed_imgs, "brisk")
        kpt_data.append(kpt1)
        kpt_match_data.append(kptM)
        i = i + 1
    print(len(kpt_data))
    if bfilter == True:
        bfilter = rt.blurred_filter(kpt_data, kpt_match_data)
    else:
        bfilter = np.ones(len(imgs), dtype=int)
        print("bfilter vypnuty")
    if rfilter == True:
        rfilter = rt.repeating_filter(kpt_data, kpt_match_data)
    else:
        rfilter = np.ones(len(imgs), dtype=int)
        print("rfilter vypnuty")
    filter = []
    for i in range(0, len(bfilter)):
        if bfilter[i] == 1 and rfilter[i] == 1:
            filter.append("1")
        else:
            filter.append("0")

    return ("".join(filter))