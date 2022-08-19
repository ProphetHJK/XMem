from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import cv2
import numpy as np


def Gaussian(img,h,w):
    img = img.convert("RGB")
    # 放大到2倍
    img = img.resize(((w*2, h*2)),Image.LANCZOS)
    img = np.array(img)
    # 高斯滤波模糊边缘
    img_gaosi=cv2.GaussianBlur(img,(5,5),0)                     #高斯降噪，设置高斯核
    img = Image.fromarray(img_gaosi)                            #转换回数组，以便numpy可读取
    # 不知道干什么
    img = img.resize((w*2, h*2), Image.ANTIALIAS)
    
    # 放大到4倍
    img = img.resize(((w*4, h*4)),Image.LANCZOS)
    img = np.array(img)
    # 高斯滤波模糊边缘
    img_gaosi=cv2.GaussianBlur(img,(5,5),0)                     #高斯降噪，设置高斯核
 
    # 还原到原大小
    img = cv2.resize(img, (w, h))
    return img

def __main__():

    outdir = 'workspace/11/masks2/'

    #/path/to/output/your/directory/for/read
    imageList = sorted(glob.glob('workspace/11/masks/*.png'))

    for item in imageList:
        img_path = item       #获取图片路径
        img = Image.open(img_path)  
        img_sp = cv2.imread(img_path)   
        sp = img_sp.shape           #读取图片长宽
        temp = os.path.basename(item)
        img = Gaussian(img,sp[0],sp[1])
        
        
        # img = img.convert("P",colors=2)
        img.save(os.path.join(outdir,temp))