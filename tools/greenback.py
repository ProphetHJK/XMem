from turtle import back
import cv2
import numpy as np
import cupy as cp
import glob
import configparser,sys
from PIL import Image
import os,re
import resize,time

"""
本程序用于生成绿幕视频，方便PR等视频剪辑工具导入

"""


def convert_str_to_float(s: str) -> float:
    """Convert rational or decimal string to float
    """
    if '/' in s:
        num, denom = s.split('/')
        return float(num) / float(denom)
    return float(s)

start = time.time()

config = configparser.ConfigParser()
config.read('tools/config.ini')
src_file_name = config['config']['src_file_name']
src_file_ext = config['config']['src_file_ext']
# mask图片中非mask部分的hsv通道中的h，不能为黑色，红0，绿120，蓝240，Xmem分割单对象默认为红色
none_mask_color_hue = 0

src_file = "source/%s.%s" % (src_file_name,src_file_ext)
dst_file = 'workspace/%s/greenback.mp4' % src_file_name

# open up video
cap = cv2.VideoCapture(src_file)
red_mask_list = sorted(glob.glob('workspace/%s/masks/*.png' % src_file_name))
backimg_path = None
# backimg_path = "1.jpg"

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
# 获取帧率
outstr = "".join(os.popen("ffprobe -v quiet -show_streams -select_streams v:0 %s |grep \"r_frame_rate\"" % src_file))
framerate = re.search("r_frame_rate=(.*)",outstr).group(1)
fr = convert_str_to_float(framerate)
<<<<<<< HEAD
fourcc = cv2.VideoWriter_fourcc(*'I420')
if os.path.exists(dst_file+'.avi'):
    os.remove(dst_file+'.avi')
=======
# 文件为未压缩版本，会较大
fourcc = cv2.VideoWriter_fourcc(*'I420')
>>>>>>> 78a75dc37002630888bf501d126b8b5b09c205c8
out = cv2.VideoWriter(dst_file+'.avi',fourcc, fr, res)

# 获取总帧数
outstr = "".join(os.popen("ffprobe -v quiet -show_streams -select_streams v:0 %s |grep \"nb_frames\"" % src_file))
total_framenum = re.search("nb_frames=(.*)",outstr).group(1)
total_framenum = int(total_framenum) - 1

frame_num = 0

end = time.time() - start
print('初始化完成：用时：{}'.format(end))
start = time.time()

# loop
done = False
while not done:
    # get frame,第一帧之前读过了,跳过读取
    if frame_num != 0:
        ret, img = cap.read()
    else:
        ret = 1
        img = frame
    if not ret:
        done = True
        continue
    print('进度：%d/%d\r' % (frame_num,total_framenum),end='')
    red_mask = red_mask_list[frame_num]
    # print(red_mask)
    # resize,考虑加上resize.py的高斯滤波
    # img = cv2.resize(img, res)
    red_mask_img_file = Image.open(red_mask)
    # red_mask_img = resize.Gaussian(red_mask_img_file, h, w)
    # 暂时先别用高斯模糊
    red_mask_img = red_mask_img_file.convert("RGB")
    red_mask_img = np.array(red_mask_img)
    # TODO:mask图片拉伸后会有锯齿，可以尝试消除
    red_mask_img = cv2.resize(red_mask_img, (w, h))
    red_mask_img_file.close()

    # change to hsv
    hsv = cv2.cvtColor(red_mask_img, cv2.COLOR_RGB2HSV)
    hue,s,v = cv2.split(hsv)


    # 提取红色构建mask矩阵，处理后mask矩阵中红色为0，非红色为255，和像素点一一对应
    # 第一个参数：原始值
    # 第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
    # 第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
    # 而在lower_red～upper_red之间的值变成255
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
    # crop = np.zeros_like(img) # 构造大小相同全0矩阵
    # 将红色mask部分填充为原图片
    
    # crop = cp.asarray(img)
    crop = img

    # with open("randomfile.txt", "w+") as external_file:
    #     np.set_printoptions(threshold=np.inf)
    #     print(img, file=external_file)
    #     external_file.close()
    # sys.exit()
    if backimg_path is None:
        # 将非红色mask部分填充为绿色
        crop[mask == 0] = [0,255,0] 
    else: 
        backimg = cp.asarray(backimg)
        # 将非红色mask部分填充为背景
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
    # crop = cp.asnumpy(crop)

    out.write(crop)

    frame_num  = frame_num + 1


end = time.time() - start
print('avi生成完成：用时：{}'.format(end))
start = time.time()

# close caps
cap.release()
out.release()

os.system('ffmpeg -i %s -i %s -map 0:v -map 1:a? -c:v libx265 -crf 26 -c:a copy %s' % (dst_file+'.avi',src_file, dst_file))
    
end = time.time() - start
print('mp4生成完成：用时：{}'.format(end))
start = time.time()
