# 配套工具

## Xmem的使用介绍

### 首先安装必要的依赖

- python3.8
- pythorch(包括torchvision，<https://pytorch.org/>)
- OpenCV(pip install opencv-python)
- 其他(pip install -r requirements.txt)
- GUI相关(pip install -r requirements_demo.txt)

### 下载模型

```cmd
scripts/download_models_demo.sh
```

也可自行去github相关页面下载

### 素材准备

新建source文件夹，将视频放入source文件夹

### GUI打开

建议GUI，命令行看不到mask，不能实时调整。

```cmd
python interactive_demo.py --video source/1.mp4 --num_objects 1 --size 480
```

参数：

- num_objects：对象数，默认就是1
- size：处理的视频的大小，会将待处理视频自动转为该大小，-1表示原大小，建议默认480或更小，太大处理不动。缺点是生成的mask放大有锯齿
- 其他默认即可

### 打标记

需要对第一帧手动打标记，利用自带的`Click`(fbrs)和`Scribble`(s2m)、`Free`，为第一帧打上物体标记，点击`Forward Propagate`自动生成后续标记

如果发现标记错误，先点击`Clear memory`，为错误的帧重新打标记，然后重新点`Forward Propagate`自动生成后续标记或`Backward Propagate`自动生成之前的标记

生成的mask放在workspace目录，重新打开GUI会自动加载之前的标记

## 自己做的工具介绍

方便pr等视频软件导入

### mask颜色转换

```
pngcolor.py
```

用于将生成的mask的颜色转为绿底

### 生成视频

```
convertmkv.sh
```

利用ffmpeg将mask转为视频

### 重命名文件

```
rename.py
```

用于测试文件的重命名
