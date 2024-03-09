import re
import pandas as pd

# 用于存储每个epoch的数据
data = []

# 从文件中读取文本内容
with open('D:\\Unet\\unet\\Fire_Seg_weight\\best_model_UNet_Skip_enhance_two.txt', 'r') as file:
    text = file.read()

# 使用正则表达式来提取每个epoch块
epoch_blocks = re.finditer(r'\[epoch: (\d+)\]\ntrain_loss: ([0-9.]+)\nlr: ([0-9.]+)\ndice coefficient: ([0-9.]+)\nglobal correct: ([0-9.]+)\naverage row correct: \[\'([0-9.]+)\', \'([0-9.]+)\'\]\nIoU: \[\'([0-9.]+)\', \'([0-9.]+)\'\]\nmean IoU: ([0-9.]+)', text)

for match in epoch_blocks:
    epoch, train_loss, lr, dice_coefficient, global_correct, avg_row_correct_1, avg_row_correct_2, iou_1, iou_2, mean_iou = match.groups()
    data.append([epoch, train_loss, lr, dice_coefficient, global_correct, avg_row_correct_1, avg_row_correct_2, iou_1, iou_2, mean_iou])

# 创建一个DataFrame
df = pd.DataFrame(data, columns=['Epoch', 'Train Loss', 'LR', 'Dice Coefficient', 'Global Correct', 'Avg Row Correct 1',
                                 'Avg Row Correct 2', 'IoU 1', 'IoU 2', 'Mean IoU'])

# 保存DataFrame到Excel文件
df.to_excel('D:\\Unet\\unet\\Fire_Seg_weight\\best_model_UNet_Skip_enhance_two.xlsx', index=False)
