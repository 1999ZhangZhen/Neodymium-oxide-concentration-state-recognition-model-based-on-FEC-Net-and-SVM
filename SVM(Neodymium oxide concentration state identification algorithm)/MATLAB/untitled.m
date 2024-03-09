% 清空环境变量
warning off
close all
clear
clc

% 创建时间命名文件夹
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, currentDateTime);
mkdir(outputFolder);

% 导入数据
res = xlsread('FEC-Net_ALL_data.xlsx');

% 划分训练集和测试集
temp = randperm(length(res));
rate = 0.80;
P_train = res(temp(1:int16(length(res) * rate)), 1:12)';
T_train = res(temp(1:int16(length(res) * rate)), 13)';
M = size(P_train, 2);

P_test = res(temp(int16(length(res) * rate) + 1:end), 1:12)';
T_test = res(temp(int16(length(res) * rate) + 1:end), 13)';
N = size(P_test, 2);

% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
t_train = T_train;
t_test = T_test;

tongyimodelFilename = fullfile(outputFolder, 'ps_input.mat');
save(tongyimodelFilename, 'ps_input');

% 转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

% 创建模型
c = 15.0;      % 惩罚因子
g = 0.06;      % 径向基函数参数
cmd = ['-t 2', '-c', num2str(c), '-g', num2str(g)];
model = svmtrain(t_train, p_train, cmd);

% 仿真测试
T_sim1 = svmpredict(t_train, p_train, model);
T_sim2 = svmpredict(t_test, p_test, model);

% 性能评价
error1 = sum((T_sim1' == T_train)) / M * 100;
error2 = sum((T_sim2' == T_test)) / N * 100;

% 数据排序
[T_train, index_1] = sort(T_train);
[T_test, index_2] = sort(T_test);

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

% 保存模型
modelFilename = fullfile(outputFolder, 'svm_model.mat');
save(modelFilename, 'model');

%% 降维和可视化（使用 t-SNE）
disp('Performing t-SNE for visualization...');

% 将训练集和测试集合并
all_data = [p_train; p_test];
all_labels = [t_train; t_test];

% 对数据进行 t-SNE 降维
tsne_result = tsne(all_data);

% 提取训练集和测试集的 t-SNE 结果
tsne_train = tsne_result(1:M, :);
tsne_test = tsne_result(M + 1:end, :);

% 绘制 t-SNE 可视化
figure
gscatter(tsne_train(:, 1), tsne_train(:, 2), T_train, 'bgr', 'o*x');
hold on
gscatter(tsne_test(:, 1), tsne_test(:, 2), T_test, 'bgr', '+*s');
hold off
title('t-SNE Visualization of SVM Predictions');
legend('Class 0', 'Class 1', 'Class 2');  % 根据标签的实际情况更新类别名称
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');

% 保存 t-SNE 可视化图像
tsne_visualization_filename = fullfile(outputFolder, 't-SNE_Visualization.png');
saveas(gcf, tsne_visualization_filename);

%% 混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Train.png'));

figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Test.png'));
