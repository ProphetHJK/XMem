# XMem

## 说明

Xmem用于视频抠像，优点是消耗的内存和GPU资源少，且对象识别效果非常出色

## 演示

<https://www.bilibili.com/video/BV1kW4y1h7N3>

## Xmem的使用介绍

### 首先安装必要的依赖

- python3.8(官方说3.8以上都行，但实测会有些问题，建议直接3.8版本)
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

- （推荐）方式一：修改tools/config.ini，执行以下命令

    ```cmd
    python interactive_demo.py
    ```

- 方式二：手动填写参数，执行以下命令

    ```cmd
    python interactive_demo.py --model "./saves/XMem-s012.pth" --num_objects 1 --size 480 --video source/1.mp4
    ```

参数说明：

- `model`: 预训练的模型路径，见RESULTS.md
- `num_objects`：对象数，默认就是1
- `size`：处理的视频的大小，会将待处理视频自动转为该大小，-1表示原大小，建议默认480或更小（如果你的显卡够强可以直接-1），太小的缺点是生成的mask放大有锯齿
- `video`：源视频路径
- 其他默认即可

### 打标记

需要对第一帧手动打标记，利用自带的`Click`(fbrs)和`Scribble`(s2m)、`Free`，为第一帧打上物体标记，点击`Forward Propagate`自动生成后续标记

如果发现标记错误，先点击`Clear memory`，为错误的帧重新打标记，然后重新点`Forward Propagate`自动生成后续标记或`Backward Propagate`自动生成之前的标记

生成的mask放在workspace目录，重新打开GUI会自动加载之前的标记

### 进阶：多对象标记

使用数字键`0-9`切换对象，默认是1号对象，每个对象的标记颜色不同，可以用于区分

## 自己做的工具介绍

放在tools文件夹，用工具处理后方便PR、剪映等视频剪辑软件导入

### 生成绿幕视频

用于生成绿幕视频

前置要求：

- [FFmpeg](https://ffmpeg.org/download.html#build-windows)(获取源视频信息，生成目标视频)
- [Git Bash](https://gitforwindows.org/)(用到了grep命令，Linux一般自带)
- [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)(加速numpy矩阵计算)
- python3

```powershell
PS H:\XMem> python tools/greenback.py
```

生成的视频在 `workspace/{视频文件名}/greenback.mp4`

更新：已添加自动读取源文件信息并生成绿底视频功能，需要先修改tools/config.ini

### 生成mask视频

利用ffmpeg将mask图片组转为视频

前置要求：

- 同[生成绿幕视频](#生成绿幕视频)

```powershell
PS H:\XMem> python tools/pngcolor.py
```

生成的视频在 `workspace/{视频文件名}/masks/output.mkv`

目前还有BUG，遇到非固定速率视频会造成生成的mask视频和源视频不匹配，此时可以尝试将源视频转为固定帧速率

### 后续TODO

1. 优化边缘锯齿（对于性能较弱的显卡不得不降低素材分辨率，导致mask图片边缘会有锯齿）
2. 非固定速率视频的mask匹配（目前可以使用生成绿幕视频功能解决该问题，但有可能导致音画不匹配）
3. windows下的GUI工具
4. 用numpy库做视频的绿幕替换太慢了，尝试用ffmpeg解决
