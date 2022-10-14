# XMem

## 说明

Xmem 用于视频抠像，优点是消耗的内存和 GPU 资源少，且对象识别效果非常出色

## 准备

- Nvidia GTX 或 RTX 显卡(GeForce GTX 1070)
- 高性能 CPU(AMD Ryzen™ 5 5600X)
- 固态硬盘

## 演示

<https://www.bilibili.com/video/BV1kW4y1h7N3>

## Xmem 的使用介绍

### 首先安装必要的依赖

- Python3.8(官方说 3.8 以上都行，但实测会有些问题，建议直接 3.8 版本)
- pythorch(包括 torchvision，<https://pytorch.org/>)
- OpenCV(pip install opencv-python)
- 其他(pip install -r requirements.txt)
- GUI 相关(pip install -r requirements_demo.txt)

### 下载模型

```cmd
scripts/download_models_demo.sh
```

也可自行去 github 相关页面下载

### 素材准备

新建 source 文件夹，将视频放入 source 文件夹

### GUI 操作

建议 GUI，命令行看不到 mask，不能实时调整。

- 方式一：修改`tools/config.ini`，执行以下命令打开 GUI

  ```cmd
  python interactive_demo.py
  ```

### 打标记

需要对第一帧手动打标记，利用自带的`Click`(fbrs)和`Scribble`(s2m)、`Free`，为第一帧打上物体标记，点击`Forward Propagate`自动生成后续标记

如果发现标记错误，先点击`Clear memory`，为错误的帧重新打标记，然后重新点`Forward Propagate`自动生成后续标记或`Backward Propagate`自动生成之前的标记

生成的 mask 放在 workspace 目录，重新打开 GUI 会自动加载之前的标记

### 进阶：多对象标记

使用数字键`0-9`切换对象，默认是 1 号对象，每个对象的标记颜色不同，可以用于区分

## 自己做的工具介绍

放在 tools 文件夹，用工具处理后方便 PR、剪映等视频剪辑软件导入

### 生成绿幕视频

用于生成绿幕视频

前置要求：

- [FFmpeg](https://ffmpeg.org/download.html#build-windows)(获取源视频信息，生成目标视频)
- ~~[Git Bash](https://gitforwindows.org/)(用到了 grep 命令，Linux 一般自带)~~(当前版本不需要)
- [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)(加速 numpy 矩阵计算)
- python3
- cupy(注意和 CUDA 的版本匹配，否则会安装失败)

```powershell
PS H:\XMem> python tools/greenback.py
```

生成的视频在 `workspace/{视频文件名}/greenback.mp4`

更新：已添加自动读取源文件信息并生成绿底视频功能，需要先修改 tools/config.ini

### 生成 mask 视频

利用 ffmpeg 将 mask 图片组转为视频，用于 PR 剪辑使用

前置要求：

- 同[生成绿幕视频](#生成绿幕视频)

```powershell
PS H:\XMem> python tools/pngcolor.py
```

生成的视频在 `workspace/{视频文件名}/masks/output.mkv`

目前还有 BUG，遇到非固定速率视频会造成生成的 mask 视频和源视频不匹配，此时可以尝试将源视频转为固定帧速率

### 后续 TODO

1. ~~优化边缘锯齿（对于性能较弱的显卡不得不降低素材分辨率，导致 mask 图片边缘会有锯齿）~~（已完成，使用 opencv 提供的中值滤波功能消除锯齿，效果不错，缺点是吃 CPU）
2. ~~非固定速率视频的 mask 匹配（目前可以使用生成绿幕视频功能解决该问题，但有可能导致音画不匹配）~~（已完成，使用生成绿幕视频功能，经过优化速度已可接受）
3. windows 下的 GUI 工具
4. ~~用 numpy 库做视频的绿幕替换太慢了，尝试用 ffmpeg 解决~~（已完成，使用 cupy 和多线程处理充分利用 CPU 和 GPU，目前速度可接受）
