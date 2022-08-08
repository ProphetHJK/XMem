import os
path='source'
old_dir=os.listdir(path) #获取/home/linuxidc/linuxidc.com目录下的所有文件目录
# print("原始目录为 %s"%old_dir)
tmpi = 0
for i in old_dir:
    new_name= str(tmpi).zfill(6) + '.png'
    os.rename(os.path.join(path,i),os.path.join(path,new_name))
    tmpi = tmpi + 1
# new_dir=os.listdir(path)
# print("现在的目录为%s"%new_dir)