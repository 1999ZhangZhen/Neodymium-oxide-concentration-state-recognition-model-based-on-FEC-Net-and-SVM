from PIL import Image
import numpy as np
import os

# 设置阈值（可根据具体情况调整）
threshold = 128

path1 = r'resultCD'
path2 = r'result1'
if not os.path.exists(path2):
    os.makedirs(path2)

# 遍历文件夹中的所有图片
for file_name in os.listdir(path1):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        # 读取图片并转换为灰度图像
        img = Image.open(os.path.join(path1, file_name)).convert("L")

        # 将灰度图像转换为numpy数组
        arr = np.array(img)

        # 根据阈值进行二值化
        arr[arr < threshold] = 0
        arr[arr >= threshold] = 255

        # 转换回PIL Image对象并保存为PNG格式
        out_img = Image.fromarray(arr.astype(np.uint8))
        out_img.save(os.path.join(path2, file_name[:-4] + "_binary.png"))