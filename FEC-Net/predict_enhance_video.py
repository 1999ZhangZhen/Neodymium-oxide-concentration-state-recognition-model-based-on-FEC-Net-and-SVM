import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet
import cv2
from src.unet import UNet_CBAM, UNet_CBAM2, UNet_CARAFE, UNet_SPPF, UNet_CBAM_SPPF, UNet_Skip_enhance_CBAM_FasterNetBlock


def process_frame(frames):
    hsv_image = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)
    # 定义橘色火焰和熔盐的颜色范围
    orange_lower = np.array([10, 100, 100], dtype=np.uint8)
    orange_upper = np.array([25, 255, 255], dtype=np.uint8)
    # 定义白色火焰和熔盐的颜色范围
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([179, 30, 255], dtype=np.uint8)
    # 定义红色火焰和熔盐的颜色范围
    red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170, 100, 100], dtype=np.uint8)
    red_upper2 = np.array([179, 255, 255], dtype=np.uint8)

    orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    red_mask = cv2.inRange(hsv_image, red_lower1, red_upper1) + cv2.inRange(hsv_image, red_lower2, red_upper2)
    # 搜索符合条件的像素点
    orange_pixels = np.where(orange_mask == 255)
    white_pixels = np.where(white_mask == 255)
    red_pixels = np.where(red_mask == 255)
    result_mask = cv2.bitwise_or(cv2.bitwise_or(orange_mask, white_mask), red_mask)
    th1 = cv2.erode(result_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    rongyan_and_huoyan = cv2.dilate(th1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)

    return rongyan_and_huoyan

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(video_path, output_folder, origin_path, roi_mask_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件.")
        return
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(origin_path, exist_ok=True)
    os.makedirs(roi_mask_path, exist_ok=True)
    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        print(frame_count)

        ret, frame = cap.read()
        if not ret:
            break

        classes = 1  # exclude background
        # weights_path = "D:\\Unet\\unet\\100epoch_weight\\best_model_UNet_Skip_enhance_CBAM_FasterNetBlock.pth"
        weights_path = "D:\\Unet\\unet\\100epoch_weight\\best_model_UNet64_base.pth"
        frame_filename = os.path.join(origin_path, f"{frame_count:04d}.png")   # 保存原始图片
        # cv2.imwrite(frame_filename, frame)
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_filename)

        roi_mask = process_frame(frame)
        roi_mask_filename = os.path.join(roi_mask_path, f"{frame_count:04d}.png")
        # cv2.imwrite(roi_mask_filename, roi_mask)
        Image.fromarray(roi_mask).save(roi_mask_filename)

        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(video_path), f"video folder {video_path} not found."
        assert os.path.exists(roi_mask_path), f"mask folder {roi_mask_path} not found."
        assert os.path.exists(origin_path), f"origin folder {origin_path} not found."

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # Create model
        model = UNet(in_channels=3, num_classes=classes + 1, base_c=64)
        # model = UNet_Skip_enhance_CBAM_FasterNetBlock(in_channels=3, num_classes=classes + 1, base_c=32)

        # Load weights
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.to(device)


        # Load ROI mask
        # roi_img = Image.open(roi_mask_path).convert('L')
        roi_img = np.array(roi_mask)
        # Load image
        # original_img = Image.open(image_path).convert('RGB')
        original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # From PIL image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # Expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # Enter evaluation mode
        with torch.no_grad():
            # Init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("Inference time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # Set the pixel values corresponding to the foreground to 255 (white)
            prediction[prediction == 1] = 255
            # Set the pixels in uninterested areas to 0 (black)
            prediction[roi_img == 0] = 0
            # result_filename = os.path.join(output_folder, image_filename)   # 保存为.tif的时候打开
            result_filename = os.path.join(output_folder, f"{frame_count:04d}.png")
            mask = Image.fromarray(prediction)
            mask.save(result_filename)

    cap.release()

if __name__ == '__main__':
    video_path = "D:\\Unet\\unet\\result_predict_avi_picture_txt\\video\\zhengchang_liao\\1087正常料.avi"  # 替换成你的视频文件路径
    output_folder = "D:\\Unet\\unet\\UNet_picture\\zhengchang_liao\\zhengchangliao1087\\predict_img"  # 保存预测结果的文件夹
    origin_path = "D:\\Unet\\unet\\UNet_picture\\zhengchang_liao\\zhengchangliao1087\\origin"  # 保存原图的文件夹
    roi_mask_path = "D:\\Unet\\unet\\UNet_picture\\zhengchang_liao\\zhengchangliao1087\\mask"  # 保存ROI掩膜的文件夹
    main(video_path, output_folder, origin_path, roi_mask_path)
