# import cv2
# import os
#
# def median_filter_and_save(input_folder, output_folder):
#     # 创建输出文件夹
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 获取输入文件夹中的所有图片文件
#     image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
#
#     for image_file in image_files:
#         # 读取图片
#         image_path = os.path.join(input_folder, image_file)
#         img = cv2.imread(image_path)
#
#         # 应用中值滤波
#         median_filtered = cv2.medianBlur(img, 5)  # 第二个参数是滤波器的大小，可以根据需要调整
#
#         # 构造输出文件路径
#         output_path = os.path.join(output_folder, f"median_{image_file}")
#
#         # 保存中值滤波后的图片
#         cv2.imwrite(output_path, median_filtered)
#
# if __name__ == "__main__":
#     input_folder = "D:\\Unet\\unet\\picture\\liao_duo\\1076\\predict_img"
#     output_folder = "D:\\Unet\\unet\\picture\\liao_duo\\1076\\zhozhi_predict_img"
#
#     median_filter_and_save(input_folder, output_folder)



import cv2
import os
import numpy as np

def median_and_morphological_opening_and_save(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for image_file in image_files:
        # 读取图片
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        # 应用中值滤波
        median_filtered = cv2.medianBlur(img, 5)  # 第二个参数是滤波器的大小，可以根据需要调整

        # 应用形态学开运算
        kernel = np.ones((5, 5), np.uint8)
        morph_opened = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)

        # 构造输出文件路径
        output_path = os.path.join(output_folder, f"median_and_opening_{image_file}")

        # 保存中值滤波和形态学开运算后的图片
        cv2.imwrite(output_path, morph_opened)

if __name__ == "__main__":
    input_folder = "D:\\Unet\\unet\\UNet_picture\\liao_duo\\liaoduo2\\predict_img"
    output_folder = "D:\\Unet\\unet\\UNet_picture\\liao_duo\\liaoduo2\\median_and_opening_predict_img"

    median_and_morphological_opening_and_save(input_folder, output_folder)
