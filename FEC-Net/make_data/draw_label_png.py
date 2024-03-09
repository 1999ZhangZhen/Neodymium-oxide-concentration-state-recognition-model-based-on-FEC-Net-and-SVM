import os
from PIL import Image
import numpy as np

# 文件路径处理
root = os.getcwd()
before = os.path.join(root, "before")
output = os.path.join(root, "output")
assert (os.path.exists(before)), "please check before folder"
assert (os.path.exists(output)), "please check output folder"
jpg = os.path.join(root, "jpg")
png = os.path.join(root, "png")
if not os.path.exists(jpg):
    os.mkdir(jpg)
if not os.path.exists(png):
    os.mkdir(png)


def main():
    # 读取原文件夹
    count = os.listdir("./before/")
    for i in range(0, len(count)):
        # 如果里的文件以jpg结尾
        # 则寻找它对应的png
        if count[i].endswith("png"):
            path = os.path.join("./before", count[i])
            img = Image.open(path)

            # 对应原图像未标注 则遍历下一张图片
            if not os.path.exists("./output/" + count[i].split(".")[0] + "_json/label.png"):
                continue

            img.save(os.path.join("./jpg", count[i]))

            # 找到对应的png
            path = "./output/" + count[i].split(".")[0] + "_json/label.png"
            img = Image.open(path)

            # 找到全局的类
            class_txt = open("./before/class_name.txt", "r")
            class_name = class_txt.read().splitlines()
            # ["bk","cat","dog"] 全局的类
            # 打开x_json文件里面存在的类，称其为局部类
            with open("./output/" + count[i].split(".")[0] + "_json/label_names.txt", "r") as f:
                names = f.read().splitlines()
                # ["bk","dog"] 局部的类
                # 新建一张空白图片, 单通道
                new = Image.new("P", (img.width, img.height))

                # 找到局部的类在全局中的类的序号
                for name in names:
                    # index_json是x_json文件里存在的类label_names.txt，局部类
                    index_json = names.index(name)
                    # index_all是全局的类,
                    index_all = class_name.index(name)

                    # 将局部类转换成为全局类
                    # 将原图img中像素点的值为index_json的像素点乘以其在全局中的像素点的所对应的类的序号 得到 其实际在数据集中像素点的值
                    # 比如dog,在局部类（output/x_json/label_names）中它的序号为1,dog在原图中的像素点的值也为1.
                    # 但是在全局的类（before/classes.txt）中其对应的序号为2，所以在新的图片中要将局部类的像素点的值*全局类的序号，从而得到标签文件
                    new = new + (index_all * (np.array(img) == index_json))

            new = Image.fromarray(np.uint8(new))
            # 将转变后的得到的新的最终的标签图片保存到make_dataset/png文件夹下
            new.save(os.path.join("./png", count[i].replace("jpg", "png")))
            # 找到新的标签文件中像素点值的最大值和最小值，最大值为像素点对应的类在class_name.txt中的序号，最小值为背景，即0
            print(np.max(new), np.min(new))


if __name__ == '__main__':
    main()
