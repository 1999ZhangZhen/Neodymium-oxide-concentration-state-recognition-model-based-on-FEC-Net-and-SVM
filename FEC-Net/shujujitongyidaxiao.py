from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(1024, 768), fill_color=(0, 0, 0)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_path):
            try:
                # 打开图像文件
                img = Image.open(input_path)

                # 创建一个新的图像对象，以指定的尺寸和颜色填充
                new_img = Image.new("RGB", target_size, fill_color)

                # 计算缩放比例
                width_ratio = target_size[0] / img.width
                height_ratio = target_size[1] / img.height
                min_ratio = min(width_ratio, height_ratio)

                # 计算缩放后的新尺寸
                new_width = int(img.width * min_ratio)
                new_height = int(img.height * min_ratio)

                # 计算粘贴位置
                paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)

                # 缩放并粘贴图像
                img = img.resize((new_width, new_height), Image.ANTIALIAS)
                new_img.paste(img, paste_position)

                # 保存结果图像
                new_img.save(output_path)

                print(f"图片 {filename} 处理完成")

            except Exception as e:
                print(f"处理图片 {filename} 时出现错误: {str(e)}")

# 指定输入和输出文件夹
input_folder = "D:\\Unet\\unet\\BoWFireDataset\\training\\mask"
output_folder = "D:\\Unet\\unet\\BoWFireDataset\\training\\re_mask"

# 执行图片转换
resize_images(input_folder, output_folder)
