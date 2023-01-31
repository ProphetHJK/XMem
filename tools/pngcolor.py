# -*- coding:utf-8 -*-
#! python3
from PIL import Image
import os,sys,re
import struct
from zlib import crc32
import shutil
import configparser


''' 将png转为黑底'''
pngsig = b'\x89PNG\r\n\x1a\n'
def swap_palette(filename):
    # open in read+write mode
    with open(filename, 'r+b') as f:
        f.seek(0)
        # verify that we have a PNG file
        if f.read(len(pngsig)) != pngsig:
            # raise RuntimeError('not a png file!')
            print("not a png file!")
            return

        while True:
            chunkstr = f.read(8)
            if len(chunkstr) != 8:
                # end of file
                break

            # decode the chunk header
            length, chtype = struct.unpack('>L4s', chunkstr)
            # we only care about palette chunks
            if chtype == b'PLTE':
                curpos = f.tell()
                paldata = f.read(length)
                # change the 3rd palette entry to cyan
                # paldata = paldata[:6] + b'\x00\xff\xde' + paldata[9:]
                # # 第一个由黑色改为绿色，第二个由暗红色改为红色
                # paldata = b'\x00\xff\x00' + b'\xff\x00\x00' + b'\x00\x00\xff' + paldata[9:]
                # 第一个由绿色改为黑色，第二个由暗红色改为红色
                paldata = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00' + paldata[12:]

                # go back and write the modified palette in-place
                f.seek(curpos)
                f.write(paldata)
                f.write(struct.pack('>L', crc32(chtype+paldata)&0xffffffff))
            else:
                # skip over non-palette chunks
                f.seek(length+4, os.SEEK_CUR)

config = configparser.ConfigParser()
config.read('tools/config.ini')
src_file_name = config['config']['src_file_name']
src_file_ext = config['config']['src_file_ext']
path='workspace/%s/' % src_file_name
src_path = path + 'masks/'
# dst_path = path + 'masks2/'
dst_path = path + 'masks/'
# if not os.path.exists(dst_path):
#     os.makedirs(dst_path)

for root, dirs, files in os.walk(src_path):
    for file in files:
        shutil.copyfile(root+file, dst_path+file)
        # PNG为P模式，非RGB模式，所以直接修改调色板
        swap_palette(dst_path+file)


''' 生成mask视频，现在直接用greenback.py就行，不需要这个了'''
# # TODO:有BUG，遇到非固定速率视频会造成不匹配
# outstr = "".join(os.popen("ffprobe -v quiet -show_streams -select_streams v:0 source/%s.%s |grep \"r_frame_rate\"" % (src_file_name,src_file_ext)))
# framerate = re.search("r_frame_rate=(.*)",outstr).group(1)
# print(framerate)
# os.system("ffmpeg -framerate %s -y -i %s%%07d.png -c:v copy %soutput.mkv" % (framerate, dst_path,dst_path))