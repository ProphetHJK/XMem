# 直接打包成mkv
ffmpeg -framerate 30 -i %07d.png -c:v copy output.mkv
# 使用无压缩方式生成h264编码的mp4
ffmpeg -framerate 30 -i %07d.png -c:v libx264rgb -crf 0 output.mp4
# 如果输入文件是图片，会自动使用image2模式，可省略image2
ffmpeg -threads 2 -f image2 -r 30 -i %07d.png output.mp4