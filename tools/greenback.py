import cv2
import numpy as np
import glob
import configparser,sys
from PIL import Image
import os,re
import resize,time
import ffmpeg
import threading

cupyflag = True
try:
    import cupy as cp
except ImportError:
    print("Can't find cupy，see readme.md for more imformation.")
    cupyflag = False

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

def print_all_numpy(n,file):
    np.set_printoptions(threshold=np.inf)
    with open(file, "w+") as external_file:
        print(n, file=external_file)

sem = threading.Semaphore(3)
write_sem = threading.Semaphore(1)
def apply_mask(red_mask_list,frame_num,img,backimg,out_crop_list,cupyflag):
    # sem.acquire()
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
    red_mask_img = cv2.resize(red_mask_img, (w, h),cv2.INTER_CUBIC)
    red_mask_img_file.close()

    # change to hsv
    hsv = cv2.cvtColor(red_mask_img, cv2.COLOR_RGB2HSV)
    hue,s,v = cv2.split(hsv)

    # 提取绿色（h=60）构建mask矩阵，处理后mask矩阵中绿色为255，非绿色为0，和像素点一一对应
    # 第一个参数：原始值
    # 第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
    # 第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
    # 而在lower_red～upper_red之间的值变成255
    mask = cv2.inRange(hue, 30, 67)
    # mask = cv2.inRange(s, 0, 0)
    # print_all_numpy(mask,'beforesmooth.txt')
    # smooth out the mask and invert，柔化边缘，之前高斯过了，可以不用
    # kernel = np.ones((3,3), np.uint8)
    # # mask = cv2.dilate(mask, kernel, iterations = 1)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.medianBlur(mask, 3)
    # # mask = cv2.bitwise_not(mask)
    # print_all_numpy(mask,'aftersmooth.txt')
    # cv2.imshow("Mask", mask)
    # if cv2.waitKey(1000) == ord('q'):
    #     sys.exit()

    if cupyflag:
        crop = cp.asarray(img)
    else:
        crop = img

    if backimg is None:
        # 将绿色mask部分填充为绿色
        crop[mask == 255] = [0,255,0] 
    else: 
        # 将绿色mask部分填充为背景
        rows, cols = crop.shape[:2]
        for i in range(rows):
            for j in range(cols):
                if mask[i,j] == 255:
                    crop[i, j] = backimg[i, j]

    # show
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Blank", crop)
    # cv2.imshow("Image", img)
    # done = cv2.waitKey(1) == ord('q')

    # save
    if cupyflag:
        crop = cp.asnumpy(crop)
    out_crop_list[frame_num] = crop
    sem.release()

max_frame_num = None
def write_out_crop(out,out_crop_list):
    now_num = 0
    while True:
        try:
            out.write(out_crop_list[now_num])
        except KeyError as ex:
            if max_frame_num == now_num:
                break
            else:
                write_sem.acquire()
                # time.sleep(0.01)
        else:
            out_crop_list.pop(now_num)
            now_num = now_num + 1

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
    if cupyflag:
        backimg = cp.asarray(backimg)

# videowriter 
res = (w, h)
# 获取帧率,TODO:用r_frame_rate还是avg_frame_rate
videoinfo = ffmpeg.probe(src_file)
vs = next(c for c in videoinfo['streams'] if c['codec_type'] == 'video')
framerate = vs['r_frame_rate']
total_framenum = vs['nb_frames']
# old
# outstr = "".join(os.popen("ffprobe -v quiet -show_streams -select_streams v:0 %s |grep \"r_frame_rate\"" % src_file))
# framerate = re.search("r_frame_rate=(.*)",outstr).group(1)

fr = convert_str_to_float(framerate)
fourcc = cv2.VideoWriter_fourcc(*'I420')
if os.path.exists(dst_file+'.avi'):
    os.remove(dst_file+'.avi')
out = cv2.VideoWriter(dst_file+'.avi',fourcc, fr, res)

# 获取总帧数
# outstr = "".join(os.popen("ffprobe -v quiet -show_streams -select_streams v:0 %s |grep \"nb_frames\"" % src_file))
# total_framenum = re.search("nb_frames=(.*)",outstr).group(1)

max_framenum = int(total_framenum) - 1

frame_num = 0

end = time.time() - start
print('初始化完成：用时：{}'.format(end))
start = time.time()

out_crop_list = {}

write_thread = threading.Thread(target=write_out_crop, args=(out,out_crop_list))
write_thread.start()

# loop
while True:
    ret = 1
    # get frame
    if frame_num != 0:
        ret, img = cap.read()
    else: 
        # 第一帧之前读过了,跳过读取
        img = frame
    
    if not ret:
        # 全部读取完成
        max_frame_num = frame_num
        write_sem.release()
        print('All frames have been read,break now')
        break
    # if frame_num >= len(red_mask_list):
    if frame_num >= 500:
        max_frame_num = frame_num
        write_sem.release()
        print('Src video length longer than mask,break now')
        break
    print('%d,进度：%d/%d\r' % (len(out_crop_list),frame_num,max_framenum),end='')
    thread = threading.Thread(target=apply_mask, args=(
                    red_mask_list,frame_num,img,None,out_crop_list,cupyflag))
    sem.acquire()
    thread.start()
    if frame_num % 3 == 0:
        write_sem.release()
    if len(out_crop_list) > 10:
        time.sleep(0.1)
    frame_num  = frame_num + 1

write_thread.join()

end = time.time() - start
print('avi生成完成：用时：{}'.format(end))
start = time.time()

# close caps
cap.release()
out.release()

#  -shortest 用于音频长于视频时缩短音频长度使两者等长，libx265 crf 26质量基本不会损失画质且视频占用空间能缩小很多
os.system('ffmpeg -i %s -i %s -map 0:v -map 1:a? -c:v libx265 -crf 26 -c:a copy -shortest %s' % (dst_file+'.avi',src_file, dst_file))
    
end = time.time() - start
print('mp4生成完成：用时：{}'.format(end))
start = time.time()
