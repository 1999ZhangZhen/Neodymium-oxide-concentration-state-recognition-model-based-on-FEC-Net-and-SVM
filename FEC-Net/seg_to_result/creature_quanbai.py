from PIL import Image
import os

def process_images(input_folder, output_folder):
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 检查文件是否是图片
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 打开图片
            image = Image.open(input_path)

            # 获取图片尺寸
            width, height = image.size

            # 创建一个与图片尺寸相同的白色图片
            white_image = Image.new('L', (width, height), color=255)

            # 保存白色图片
            white_image.save(output_path)

            print(f'Processed: {filename}')

if __name__ == "__main__":
    input_folder = "D:\\Unet\\unet\\seg_to_result\\result1"  # 替换为你的输入文件夹路径
    output_folder = "D:\\Unet\\unet\\seg_to_result\\quanbai"  # 替换为你的输出文件夹路径

    process_images(input_folder, output_folder)
