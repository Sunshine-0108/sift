# 图像拼接stitcher方法
import cv2 as cv
import numpy as np
import os
import re
import time

def cv_imread(file_path):
    img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图像: {file_path}")
    return img

# 按数字顺序排序
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

starttime = time.time()
imgs = []

# 图片文件夹路径
folder_path = 'try9_rgb'


files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

files.sort(key=extract_number)

# 遍历排序后的文件
for filename in files:
    img_path = os.path.join(folder_path, filename)
    # 读取拼接图片
    image = cv_imread(img_path)
    if image is not None:
        print(f"读取图像: {img_path}, 图像维度: {image.shape}")
        imgs.append(image)
    else:
        print(f"读取图像失败: {img_path}")

if len(imgs) < 3:
    print("图像数量不足，无法进行拼接。")
else:
    # 拼接成全景图
    stitcher = cv.Stitcher.create()
    (status, pano) = stitcher.stitch(imgs)

    # 拼接结果
    if status != cv.Stitcher_OK:
        print(f"不能拼接图片, error code = {status}")
    else:
        print("拼接成功.")
        # 输出图片
        cv.imwrite("try9_rgb/stitcher_finished.jpg", pano)
        endtime = time.time()
        print(f"拼接时间 {endtime - starttime} 秒!")
        cv.imshow('pano', pano)
        cv.waitKey(0)
