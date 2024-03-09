import os
import time

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet
from src.unet import  UNet_Skip_enhance_CBAM_FasterNetBlock, UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "D:\\Unet\\unet\\100epoch_weight\\best_model_UNet_Skip_enhance_CBAM_FasterNetBlock.pth"
    input_folder = "./DRIVE/test/images/"
    roi_mask_folder = "./DRIVE/test/mask/"
    output_folder = "./zzzzz/resume_mydata_999/train/"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_folder), f"input folder {input_folder} not found."
    assert os.path.exists(roi_mask_folder), f"mask folder {roi_mask_folder} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    # mean = (0.48872365, 0.11469327, 0.05210163)
    # std = (0.08026049, 0.02123111, 0.01563618)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Create model
    # model = UNet(in_channels=3, num_classes=classes + 1, base_c=64)
    model = UNet_Skip_enhance_CBAM_FasterNetBlock(in_channels=3, num_classes=classes + 1, base_c=32)


    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for image_filename in os.listdir(input_folder):
        if image_filename.endswith(".tif"):
            image_path = os.path.join(input_folder, image_filename)
            roi_mask_filename = image_filename.replace(".tif", "_mask.gif")
            roi_mask_path = os.path.join(roi_mask_folder, roi_mask_filename)

            # Check if the corresponding mask file exists
            if not os.path.exists(roi_mask_path):
                print(f"Mask not found for {image_filename}. Skipping.")
                continue

            # Load ROI mask
            roi_img = Image.open(roi_mask_path).convert('L')
            roi_img = np.array(roi_img)

            # Load image
            original_img = Image.open(image_path).convert('RGB')

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
                result_filename = os.path.join(output_folder, image_filename.replace(".tif", ".png"))
                mask = Image.fromarray(prediction)
                mask.save(result_filename)


if __name__ == '__main__':
    main()
