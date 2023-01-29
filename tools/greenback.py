import cv2
import numpy as np
import glob
import configparser,sys
from PIL import Image
import os,re
import resize,time
import ffmpeg
import threading
import subprocess as sp
import shlex

cupyflag = True
try:
    import cupy as cp
except ImportError:
    print("Can't find cupy，see readme.md for more imformation.")
    cupyflag = False

"""
本程序用于生成绿幕视频，方便PR等视频剪辑工具导入

"""
# 使用opencv提供的VideoWriter功能，否则使用ffmpeg
videowriter_flag = False

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

# 多线程处理mask,根据电脑实际情况配置，videowriter是瓶颈就减少，mask处理是瓶颈就增加
sem = threading.Semaphore(8)
# 图片写入视频线程所用队列的信号
write_sem = threading.Semaphore(1)
# 写入完成信号
end_sem = threading.Semaphore(0)
def apply_mask(red_mask_list,frame_num,img,w,h,backimg,out_crop_list,cupyflag,first_mask_filename_int,img_green,object_num):
    frame_num_tmp = frame_num
    frame_num = frame_num - first_mask_filename_int
    red_mask = red_mask_list[frame_num]
    red_mask_filename = os.path.splitext(os.path.basename(red_mask))[0]
    red_mask_filename_int = int(red_mask_filename)
    if frame_num < 0:
        crop = img_green
    elif red_mask_filename_int != frame_num_tmp:
        print('\nError,frame_num error,red_mask_filename_int:%d,frame_num:%d\n' % (red_mask_filename_int,frame_num))
        sys.exit()
    else:
        red_mask_img = cv2.imread(red_mask)
        # 大于一种颜色时要先合并成一种颜色
        if object_num > 1:
            green_mask = red_mask_img[:,:,0] + (red_mask_img[:,:,1] // 2)  + red_mask_img[:,:,2]
            red_mask_img[:,:,0][green_mask != 127] = 255    #R
            red_mask_img[:,:,1][green_mask != 127] = 0    #G
            red_mask_img[:,:,2][green_mask != 127] = 0  #B
            
        # TODO:mask图片拉伸后会有锯齿，可以尝试消除
        red_mask_img = cv2.resize(red_mask_img, (w, h),cv2.INTER_CUBIC)
        
        # 使用腐蚀加扩展消除噪点，效果不好，暂时不用
        # kernel = np.ones((3,3), np.uint8)
        # red_mask_img = cv2.dilate(red_mask_img, kernel, iterations = 1)
        # mask = cv2.erode(mask, kernel, iterations = 1)
        # 使用中值滤波消除锯齿，效果不错，但很吃cpu
        red_mask_img = cv2.medianBlur(red_mask_img, 21)

        # 转为HSV比较好找邻近色，mask会更平滑
        hsv = cv2.cvtColor(red_mask_img, cv2.COLOR_RGB2HSV)
        hue,s,v = cv2.split(hsv)

        # 提取绿色（h=60）构建mask矩阵，处理后mask矩阵中绿色为255，非绿色为0，和像素点一一对应
        # 第一个参数：原始值
        # 第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
        # 第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
        # 而在lower_red～upper_red之间的值变成255
        # 临近色30，60-30，60+30
        mask = cv2.inRange(hue, 30, 90)

        # 一些调试打印信息
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

        # save
        if cupyflag:
            crop = cp.asnumpy(crop)
    out_crop_list[frame_num_tmp] = crop
    sem.release()

max_frame_num = None

def write_out_crop(out,out_crop_list):
    now_num = 0
    while True:
        try:
            if videowriter_flag == True:
                out.write(out_crop_list[now_num])
            else:
                out.stdin.write(out_crop_list[now_num].astype(np.uint8).tobytes())
        except KeyError as ex:
            if max_frame_num == now_num:
                break
            else:
                # 队列里没有要处理的就先阻塞
                write_sem.acquire()
        else:
            out_crop_list.pop(now_num)
            now_num = now_num + 1
    end_sem.release()

start = time.time()

config = configparser.ConfigParser()
config.read('tools/config.ini',encoding='utf8')
src_file_name = config['config']['src_file_name']
src_file_ext = config['config']['src_file_ext']
object_num = config['config']['object_num']
object_num = int(object_num)
# mask图片中非mask部分的hsv通道中的h，不能为黑色，红0，绿120，蓝240，Xmem分割单对象默认为红色
none_mask_color_hue = 0

src_file = "source/%s.%s" % (src_file_name,src_file_ext)
dst_file = 'workspace/%s/greenback.mp4' % src_file_name

# open up video
cap = cv2.VideoCapture(src_file)
red_mask_list = sorted(glob.glob('workspace/%s/masks/*.png' % src_file_name))
backimg_path = None
# backimg_path = "1.jpg"

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

if os.path.exists(dst_file):
    os.remove(dst_file)
if videowriter_flag == True:
    fr = convert_str_to_float(framerate)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if os.path.exists(dst_file+'.avi'):
        os.remove(dst_file+'.avi')
    out = cv2.VideoWriter(dst_file+'.avi',fourcc, fr, res)
else:
    # Open ffmpeg application as sub-process
    # FFmpeg input PIPE: RAW images in BGR color format
    # FFmpeg output MP4 file encoded with HEVC codec.
    # Arguments list:
    # -y                   Overwrite output file without asking
    # -s {width}x{height}  Input resolution width x height (1344x756)
    # -pixel_format bgr24  Input frame color format is BGR with 8 bits per color component
    # -f rawvideo          Input format: raw video
    # -r {fps}             Frame rate: fps (25fps)
    # -i pipe:             ffmpeg input is a PIPE
    # -vcodec libx265      Video codec: H.265 (HEVC)
    # -pix_fmt yuv420p     Output video color space YUV420 (saving space compared to YUV444)
    # -crf 24              Constant quality encoding (lower value for higher quality and larger output file).
    # {output_filename}    Output file name: output_filename (output.mp4)
    # 如遇错误：Picture width must be an integer multiple of the specified chroma subsampling，是指yuv420p格式下视频长宽必须是2（或4）的倍数，源图片需要缩放大小
    out = sp.Popen(shlex.split(f'ffmpeg -y -loglevel warning -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {framerate} -thread_queue_size 32 -i pipe: -i {src_file} -map 0:v -map 1:a? -c:v libx265 -r {framerate} -pix_fmt yuv420p -crf 26 -c:a copy -shortest {dst_file}'), stdin=sp.PIPE)

max_framenum = int(total_framenum) - 1

frame_num = 0

# 第一张mask的序号，有时候不是从0开始的
first_mask = red_mask_list[0]
first_mask_filename = os.path.splitext(os.path.basename(first_mask))[0]
first_mask_filename_int = int(first_mask_filename)

# 创建纯绿色图片
img_green = np.zeros([h, w, 3], np.uint8)
img_green[:, :, 1] = np.zeros([h, w]) + 255

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
    if frame_num >= len(red_mask_list):
    # if frame_num >= 20:
        max_frame_num = frame_num
        write_sem.release()
        print('Src video length longer than mask,break now')
        break
    print('进度：%d/%d,队列:%d.\r' % (frame_num,max_framenum,len(out_crop_list)),end='')
    thread = threading.Thread(target=apply_mask, args=(
                    red_mask_list,frame_num,img,w,h,None,out_crop_list,cupyflag,first_mask_filename_int,img_green,object_num))
    sem.acquire()
    thread.start()
    write_sem.release()
    # 队列里待处理的过多，暂停一下
    if len(out_crop_list) > 10:
        time.sleep(0.1)
    frame_num  = frame_num + 1

# 等待帧全部写入完成
while end_sem._value == 0:
    write_sem.release()
    time.sleep(0.01)

write_thread.join()

end = time.time() - start
print('avi生成完成：用时：{}'.format(end))
start = time.time()

# close caps
cap.release()
if videowriter_flag == True:
    out.release()
    #  -shortest 用于音频长于视频时缩短音频长度使两者等长，libx265 crf 26质量基本不会损失画质且视频占用空间能缩小很多
    os.system('ffmpeg -i %s -i %s -map 0:v -map 1:a? -c:v libx265 -crf 26 -c:a copy -shortest %s' % (dst_file+'.avi',src_file, dst_file))
else:
    out.stdin.close()
    out.wait()
    # out.terminate()

end = time.time() - start
print('mp4生成完成：用时：{}'.format(end))
start = time.time()