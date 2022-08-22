from turtle import back
import cv2
import numpy as np
import glob
import configparser,sys
from PIL import Image

import resize

config = configparser.ConfigParser()
config.read('tools/config.ini')
src_file_name = config['config']['src_file_name']
src_file_ext = config['config']['src_file_ext']
green_mask = True

# open up video
cap = cv2.VideoCapture("source/%s.mp4" % src_file_name)
red_mask_list = sorted(glob.glob('workspace/%s/masks/*.png' % src_file_name))
backimg_path = "workspace/%s/masks/0000000.png" % src_file_name

# 邻近色
margin = 5

# grab one frame
scale = 1
_, frame = cap.read()
h,w = frame.shape[:2]
h = int(h*scale)
w = int(w*scale)

if backimg_path is not None:
    backimg_file = Image.open(backimg_path)
    backimg = backimg_file.convert("RGB")
    backimg_file.close()
    backimg = backimg.resize(((w, h)))
    backimg = np.array(backimg)

# videowriter 
res = (w, h)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_vid.avi',fourcc, 30.0, res)

frame_num = 0

# loop
done = False
while not done:
    # get frame,第一帧之前读过了
    if frame_num != 0:
        ret, img = cap.read()
    else:
        ret = 1
        img = frame
    if not ret:
        done = True
        continue
    red_mask = red_mask_list[frame_num]
    # print(red_mask)
    # resize,考虑加上resize.py的高斯滤波
    # img = cv2.resize(img, res)
    red_mask_img_file = Image.open(red_mask)
    # red_mask_img = resize.Gaussian(red_mask_img_file, h, w)
    # 暂时先别用高斯模糊
    red_mask_img = red_mask_img_file.convert("RGB")
    red_mask_img = np.array(red_mask_img)
    red_mask_img = cv2.resize(red_mask_img, (w, h))
    red_mask_img_file.close()

    # change to hsv
    hsv = cv2.cvtColor(red_mask_img, cv2.COLOR_BGR2HSV)
    hue,s,v = cv2.split(hsv)

    if green_mask == False:
        # get uniques
        unique_colors, counts = np.unique(s, return_counts=True)

        # sort through and grab the most abundant unique color
        # 提取画面中最多的单一颜色
        big_color = None
        biggest = -1
        for a in range(len(unique_colors)):
            if counts[a] > biggest:
                biggest = counts[a]
                big_color = int(unique_colors[a])


        # get the color mask
        mask = cv2.inRange(s, big_color - margin, big_color + margin)
    else:
        # 提取红色，处理后mask矩阵中红色为0，非红色为255，和像素点一一对应
        mask = cv2.inRange(hue, 0, 0)
        # print(mask)
        # with open("randomfile.txt", "w+") as external_file:
        #     np.set_printoptions(threshold=np.inf)
        #     print(mask, file=external_file)
        #     external_file.close()
        # sys.exit()

    # smooth out the mask and invert，柔化边缘，之前高斯过了，可以不用
    # kernel = np.ones((3,3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    # mask = cv2.medianBlur(mask, 5)
    # mask = cv2.bitwise_not(mask)

    # crop out the image，提取非红色部分
    crop = np.zeros_like(img) # 构造大小相同全0矩阵
    # 将非红色部分填充为原图片
    crop[mask == 255] = img[mask == 255]
    # with open("randomfile.txt", "w+") as external_file:
    #     np.set_printoptions(threshold=np.inf)
    #     print(img, file=external_file)
    #     external_file.close()
    # sys.exit()
    if backimg_path is None:
        # 将红色部分填充为绿色
        crop[mask == 0] = [0,255,0] 
    else: 
        # 将红色部分填充为背景
        rows, cols = crop.shape[:2]
        for i in range(rows):
            for j in range(cols):
                if mask[i,j] == 0:
                  crop[i, j] = backimg[i, j]

    # show
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Blank", crop)
    # cv2.imshow("Image", img)
    # done = cv2.waitKey(1) == ord('q')

    # save
    out.write(crop)
    
    frame_num  = frame_num + 1

# close caps
cap.release()
out.release()