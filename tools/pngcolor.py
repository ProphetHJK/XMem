# -*- coding:utf-8 -*-
#! python3
from PIL import Image
import os,sys
import struct
from zlib import crc32
import shutil
import configparser


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
                # 第一个由黑色改为绿色，第二个由暗红色改为红色
                paldata = b'\x00\xff\x00' + b'\xff\x00\x00' + b'\x00\x00\xff' + paldata[9:]

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
i = 1
j = 1
path='workspace/%s/' % src_file_name
src_path = path + 'masks/'
dst_path = path + 'masks2/'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
# old_dir=os.listdir(path)
for root, dirs, files in os.walk(src_path):
    for file in files:
        shutil.copyfile(root+file, dst_path+file)
        # 方式1，PNG为P模式，非RGB模式，所以直接修改调色板
        swap_palette(dst_path+file)

        # 方式2，下面是用转成RGB再修改每个像素点的方式，太慢了
        # img = Image.open(os.path.join(root,file))#读取系统的内照片
        # print (img.size)#打印图片大小
        # print (img.palette)
        # print (img.getpixel((4,4)))
        # width = img.size[0]#长度
        # height = img.size[1]#宽度
        # img = img.convert("RGB")
        # pix = img.load()
        # for i in range(0,width):#遍历所有长度的点
        #     for j in range(0,height):#遍历所有宽度的点
        #         # data = (img.getpixel((i,j)))#打印该图片的所有点
        #         data = pix[i,j]
                
        #         # print(pix[i,j])#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
        #         # if data > 0:
        #         #     print (data)#打印RGBA的r值

        #         if (data[0]==128 and data[1]==0 and data[2]==0):#RGBA的r值大于170，并且g值大于170,并且b值大于170
        #             img.putpixel((i,j),(255,0,0,255))
        #         else:
        #             img.putpixel((i,j),(0,177,64,255))#则这些像素点的颜色改成大红色
                    

        # # img = img.convert("RGB")#把图片强制转成RGB
        # img.save(dst_path+file)#保存修改像素点后的图片
os.system("ffmpeg -framerate 30 -i %s%%07d.png -c:v copy %soutput.mkv" % (dst_path,dst_path))