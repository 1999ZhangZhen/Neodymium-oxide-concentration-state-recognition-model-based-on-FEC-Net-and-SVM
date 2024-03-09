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
    # (原有的帧处理逻辑)
    return rongyan_and_huoyan

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def save_image(image, filename):
    try:
        cv2.imwrite(filename, image)
    except Exception as e:
        print(f"Error saving image {filename}: {e}")

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
        weights_path = "D:\\Unet\\unet\\100epoch_weight\\best_model_UNet_Skip_enhance_CBAM_FasterNetBlock.pth"
        frame_filename = os.path.join(origin_path, f"{frame_count:04d}.png")
        roi_mask_filename = os.path.join(roi_mask_path, f"{frame_count:04d}.png")

        # Save original frame
        save_image(frame, frame_filename)

        # Process frame
        roi_mask = process_frame(frame)

        # Save ROI mask
        save_image(roi_mask, roi_mask_filename)

        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(video_path), f"video folder {video_path} not found."
        assert os.path.exists(roi_mask_path), f"mask folder {roi_mask_path} not found."
        assert os.path.exists(origin_path), f"origin folder {origin_path} not found."

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        # (原有的神经网络模型加载和推断逻辑)

        result_filename = os.path.join(output_folder, f"{frame_count:04d}.png")
        mask = Image.fromarray(prediction)
        save_image(mask, result_filename)

    cap.release()

if __name__ == '__main__':
    video_path = "D:\\Unet\\unet\\result_predict_avi_picture_txt\\video\\liao_shao\\料少10_jq.mp4"
    output_folder = "D:\\Unet\\unet\\result_predict_avi_picture_txt\\picture\\liao_shao\\liaoshao10\\predict_img"
    origin_path = "D:\\Unet\\unet\\result_predict_avi_picture_txt\\picture\\liao_shao\\料少10_jq\\origin"
    roi_mask_path = "D:\\Unet\\unet\\result_predict_avi_picture_txt\\picture\\liao_shao\\料少10_jq\\mask"
    main(video_path, output_folder, origin_path, roi_mask_path)
