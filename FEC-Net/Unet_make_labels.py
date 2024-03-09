import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 高斯损失函数的定义
class GMM:
    def __init__(self, num_gaussians=3, learning_rate=0.01):
        self.num_gaussians = num_gaussians
        self.learning_rate = learning_rate
        self.bg_model = None

    def initialize(self, frame):
        self.bg_model = cv2.createBackgroundSubtractorMOG2()
        self.bg_model.apply(frame, learningRate=0.1)

    def update(self, frame):
        fg_mask = self.bg_model.apply(frame, learningRate=self.learning_rate)
        return fg_mask

    def process_frame(self, frame):
        if self.bg_model is None:
            self.initialize(frame)
        fg_mask = self.update(frame)
        return fg_mask

# 分形维数的定义
def fractual(im):   # 分形维数的计算函数,以描述图像的自相似性和复杂度。
    M = im.shape[0]
    scale = []
    Nr = []
    l = 2

    while l < (M / 2):
        r = l
        blockSizeR = r
        blockSizeC = r
        ld = (l * 1.0) / M  # For binary image, G = 2, so G/M = 1/M
        nr = 0

        for row in range(0, M, blockSizeR):
            for col in range(0, M, blockSizeC):
                row1 = row
                row2 = row1 + blockSizeR
                col1 = col
                col2 = col1 + blockSizeC
                # Extract block
                oneBlock = im[row1:row2, col1:col2]
                if np.sum(oneBlock) > 0:
                    nr = nr + 1

        Nr.append(nr)
        scale.append(M / l)
        l = l * 2

    N = np.log(Nr) / np.log(2)
    S = np.log(scale) / np.log(2)
    p = np.polyfit(S, N, 1)
    return p[0]

# 火焰圆形度的计算
def calculateCircularity(contour):        # 循环度的计算
    perimeter = cv2.arcLength(contour, True)  # 求周长
    area = cv2.contourArea(contour)     # 求面积
    circularity = 4 * np.pi * area / (perimeter**2)  # 圆
    return circularity

# 火焰轮廓粗糙度的计算
def calculateRoughness(contour):               #粗糙度的计算
    perimeter = cv2.arcLength(contour, True)   # 求周长
    hull = cv2.convexHull(contour)             # 求凸包
    hull_perimeter = cv2.arcLength(hull, True)   # 求凸包的边长
    roughness = perimeter / hull_perimeter
    return roughness

# 数据读取
cap = cv2.VideoCapture('D:\\zz_my_flame\\OPencv\正常料\\1096正常料.avi')

# 创建总文件夹
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

subdirectories = ['label_origin', 'outcome_GMM_gray_2D', 'rongyan_and_huoyan']
for subdir in subdirectories:
    subdir_path = os.path.join(output_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)
k = 0  # 用于生成文件名中的编号



# 创建ImprovedGMM对象
gmm = GMM(num_gaussians=3, learning_rate=0.01)

# cv2.getStructuringElement构造形态学使用的kernel,如腐蚀何膨胀
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

length_whole = []
area_whole = []
sum_whole = []
B_whole = []
G_whole = []
R_whole = []
gray_whole= []
area_std = []
dimensions = []
circularities = []
rough = []
Area_ratio = []    # 张震
i = 0

while cap.isOpened():
    i += 1
    k += 1
    print(i)
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = gmm.process_frame(frame)
    # 前景区域二值化，将非白色（0-244）的非前景区域（包含背景以及阴影）均设为0，前景的白色（244-255）设置为255
    fg_mask_2 = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(fg_mask_2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    dilated = fg_mask_2
    # print(fg_mask.shape)
    dilated_3D = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)   # 二值图
    GMM_yt_3 = cv2.bitwise_and(frame, dilated_3D)
    GMM_yt_3_RGB = cv2.cvtColor(GMM_yt_3, cv2.COLOR_BGR2RGB)

    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 定义橘色火焰的颜色范围
    orange_lower = np.array([10, 100, 100], dtype=np.uint8)
    orange_upper = np.array([25, 255, 255], dtype=np.uint8)
    # 定义白色火焰的颜色范围
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([179, 30, 255], dtype=np.uint8)
    # 定义红色火焰的颜色范围
    red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170, 100, 100], dtype=np.uint8)
    red_upper2 = np.array([179, 255, 255], dtype=np.uint8)
    # 根据颜色范围创建掩膜
    orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    red_mask = cv2.inRange(hsv_image, red_lower1, red_upper1) + cv2.inRange(hsv_image, red_lower2, red_upper2)
    # 搜索符合条件的像素点
    orange_pixels = np.where(orange_mask == 255)
    white_pixels = np.where(white_mask == 255)
    red_pixels = np.where(red_mask == 255)
    # 将符合条件的像素点设为白色，其余像素点设为黑色
    result_mask = cv2.bitwise_or(cv2.bitwise_or(orange_mask, white_mask), red_mask)
    th1 = cv2.erode(result_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    rongyan_and_huoyan = cv2.dilate(th1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    rongyan_and_huoyan_3 = cv2.cvtColor(rongyan_and_huoyan, cv2.COLOR_GRAY2BGR)  # 二值图
    rongyan_and_huoyan_3D = cv2.bitwise_and(frame, rongyan_and_huoyan_3)
    rongyan_and_huoyan_3D_RGB = cv2.cvtColor(rongyan_and_huoyan_3D, cv2.COLOR_BGR2RGB)
    feihei_pixels2 = np.where((np.any(rongyan_and_huoyan_3D != [0, 0, 0], axis=-1)))
    non_black_pixels_rgb2 = rongyan_and_huoyan_3D[feihei_pixels2]
    R = non_black_pixels_rgb2[:, 2]
    R_mean_value_2 = np.mean(R)
    print("R2++++++++++++",R_mean_value_2)

    feihei_pixels1 = np.where((np.any(GMM_yt_3 != [0, 0, 0], axis=-1)))
    non_black_pixels_rgb1 = GMM_yt_3[feihei_pixels1]
    R = non_black_pixels_rgb1[:, 2]
    R_mean_value_1 = np.mean(R)
    print("R1++++++++++++",R_mean_value_1)

    R_mean_value = (R_mean_value_1 + R_mean_value_2)/2
    print("R+++++++++++++",R_mean_value)

    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # cv2.CHAIN_APPROX_SIMPLE的意思是选择简化的方式存储轮廓点；cv2.RETR_EXTERNAL的意思是只选择火焰的边缘轮廓，不选择内部的轮廓
    true_contours = []
    length_ = []
    area_ = []
    circularity = []
    roughness = []
    SA = 0
    for c in contours:
        if cv2.contourArea(c) > 10:
            mask_c = np.zeros_like(dilated)  # 单色掩码
            mask_C = cv2.drawContours(mask_c, [c], -1, (255, 255, 255), thickness=cv2.FILLED)
            mask_C_3D = cv2.cvtColor(mask_C, cv2.COLOR_GRAY2BGR)
            mask_C_3D_QJ = cv2.bitwise_and(frame, mask_C_3D)

            feihei_pixels2 = np.where((np.any(mask_C_3D_QJ != [0, 0, 0], axis=-1)))
            non_black_pixels_rgb2 = mask_C_3D_QJ[feihei_pixels2]
            R2 = non_black_pixels_rgb2[:, 2]
            R2_mean_value = np.mean(R2)
            # print("R2-------------",R2_mean_value)

            if R2_mean_value >= R_mean_value:
                true_contours.append(c)
                length_.append(cv2.arcLength(c, True))
                area_.append(cv2.contourArea(c))
                circularity.append(calculateCircularity(c))
                roughness.append(calculateRoughness(c))
                SA += cv2.contourArea(c)          # 张震
            elif R2_mean_value < R_mean_value :
                GMM_yt_3[mask_C == 255] = [0, 0, 0]    # 最后的掩模(去掉熔盐和抖动的部分)
                # cv2.imshow("hahahah", GMM_yt_3)
                # cv2.waitKey()

    outcome_GMM_3D = GMM_yt_3
    median_outcome = cv2.medianBlur(outcome_GMM_3D, 5)
    outcome_GMM_3D_RGB = cv2.cvtColor(median_outcome, cv2.COLOR_BGR2RGB)

    outcome_GMM_3D_RGB_1 = cv2.cvtColor(outcome_GMM_3D, cv2.COLOR_BGR2RGB)

    outcome_GMM_gray = cv2.cvtColor(median_outcome, cv2.COLOR_RGB2GRAY)
    _, outcome_GMM_gray_2D = cv2.threshold(outcome_GMM_gray, 1, 255, cv2.THRESH_BINARY)

    outcome_GMM_gray1 = cv2.cvtColor(outcome_GMM_3D, cv2.COLOR_RGB2GRAY)
    _, outcome_GMM_gray_2D_1 = cv2.threshold(outcome_GMM_gray1, 1, 255, cv2.THRESH_BINARY)
    if not area_:
        length_whole.append(0)
        area_whole.append(0)
        sum_whole.append(0)
        gray_whole.append(0)
        B_whole.append(0)
        G_whole.append(0)
        R_whole.append(0)
        area_std.append(0)   # 标准差
        dimensions.append(0)
        circularities.append(0)
        rough.append(0)
        Area_ratio.append(0)
    else:
        length_whole.append(np.mean(length_))
        area_whole.append(np.mean(area_))
        sum_whole.append(len(area_))
        feihei_pixels = np.where((np.any(median_outcome != [0, 0, 0], axis=-1)))
        non_black_pixels_rgb = median_outcome[feihei_pixels]
        B = non_black_pixels_rgb[:,0]
        G = non_black_pixels_rgb[:,1]
        R = non_black_pixels_rgb[:,2]
        gray_feihei = np.where(outcome_GMM_gray != 0)
        gray = outcome_GMM_gray[gray_feihei]
        gray_whole.append(np.mean(gray))
        B_whole.append(np.mean(B))
        G_whole.append(np.mean(G))
        R_whole.append(np.mean(R))
        area_std.append(np.std(area_,ddof=1))
        dimensions.append(fractual(outcome_GMM_gray_2D))
        circularities.append(np.mean(circularity))
        rough.append(np.mean(roughness))
        Area_ratio.append(SA/(1280*1024))

    res1 = cv2.drawContours(frame.copy(), contours=true_contours, contourIdx=-1, color=(0, 0, 0), thickness=1)  # 画出火焰的轮廓
    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)

    plt.show()

    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    label_origin = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # axes[1, 0].imshow(label_origin)       # labels 1
    # # axes[0, 0].set_title("frame")
    # axes[0, 0].axis('off')
    #
    # axes[0, 0].imshow(res1)
    # # axes[1, 0].set_title("res1")
    # axes[1, 0].axis('off')
    #
    # axes[0, 1].imshow(GMM_yt_3_RGB)
    # # axes[0, 1].set_title("GMM_yt_3_RGB")
    # axes[0, 1].axis('off')
    #
    # axes[0, 2].imshow(outcome_GMM_3D_RGB_1)
    # # axes[0, 2].set_title("outcome_GMM_3D_RGB_1")
    # axes[0, 2].axis('off')
    #
    # axes[0, 3].imshow(outcome_GMM_3D_RGB)
    # # axes[0, 3].set_title("outcome_GMM_3D_RGB")
    # axes[0, 3].axis('off')
    #
    # axes[1, 1].imshow(dilated, cmap='gray')
    # # axes[1, 1].set_title("dilated")
    # axes[1, 1].axis('off')
    #
    # axes[1, 2].imshow(outcome_GMM_gray_2D_1, cmap='gray')
    # # axes[1, 2].set_title("outcome_GMM_gray_2D_1")
    # axes[1, 2].axis('off')
    #
    # axes[1, 3].imshow(outcome_GMM_gray_2D, cmap='gray')      # labels  2
    # # axes[1, 3].set_title("outcome_GMM_gray_2D")
    # axes[1, 3].axis('off')
    #
    # axes[0, 4].imshow(rongyan_and_huoyan, cmap='gray')       # labels  3
    # # axes[1, 3].set_title("rongyan_and_huoyan")
    # axes[0, 4].axis('off')
    #
    # axes[1, 4].imshow(rongyan_and_huoyan_3D_RGB)
    # # axes[1, 3].set_title("rongyan_and_huoyan_3D_RGB")
    # axes[1, 4].axis('off')

    plt.tight_layout()
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 保存可视化的图片
    filename_base = f"{i:04d}"  # 使用四位编号，例如 0001, 0002, ...
    label_origin_filename = os.path.join(output_dir, 'label_origin', f"{filename_base}.jpg")
    outcome_GMM_gray_2D_filename = os.path.join(output_dir, 'outcome_GMM_gray_2D', f"{filename_base}.png")
    rongyan_and_huoyan_filename = os.path.join(output_dir, 'rongyan_and_huoyan', f"{filename_base}.png")

    plt.imsave(label_origin_filename, label_origin)
    plt.imsave(outcome_GMM_gray_2D_filename, outcome_GMM_gray_2D, cmap='gray')
    plt.imsave(rongyan_and_huoyan_filename, rongyan_and_huoyan, cmap='gray')


    number = range(len(length_whole))
    def pd_toexcel(number, area_whole, length_whole, sum_whole, B_whole, G_whole,
                   R_whole, gray_whole, area_std, dimensions, circularities,
                   rough, Area_ratio, filename):  # pandas库储存数据到excel
        dfData = {  # 用字典设置DataFrame所需数据
            '序号': number,
            '平均面积': area_whole,
            '平均周长': length_whole,
            '火苗个数': sum_whole,
            'B均值': B_whole,
            'G均值': G_whole,
            'R均值': R_whole,
            '灰度均值': gray_whole,
            '尺寸标准差': area_std,
            '分形维数': dimensions,
            '火苗圆形度': circularities,
            '轮廓粗糙度': rough,
            '火焰面积占比': Area_ratio
        }
        df = pd.DataFrame(dfData)  # 创建DataFrame
        df.to_excel(filename, index=False)  # 存表，去除原始索引列（0,1,2...）

    pd_toexcel(number, area_whole, length_whole, sum_whole, B_whole, G_whole, R_whole,
               gray_whole, area_std, dimensions, circularities, rough, Area_ratio,
               'D:\\zz_my_flame\\OPencv\\阳极效应.xlsx')

cap.release()
cv2.destroyAllWindows()
