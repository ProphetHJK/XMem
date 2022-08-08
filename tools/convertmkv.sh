# 直接打包成mkv
ffmpeg -r 10 -i %06d.png -c:v copy output.mkv
# 使用无压缩方式生成h264编码的mp4
ffmpeg -r 10 -i %06d.png -c:v libx264rgb -crf 0 output.mp4