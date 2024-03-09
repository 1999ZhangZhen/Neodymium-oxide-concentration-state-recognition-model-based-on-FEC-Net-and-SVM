%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  创建时间命名文件夹
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, currentDateTime);
mkdir(outputFolder);

%%  导入数据
res = xlsread('FEC-Net_ALL_data.xlsx');
% res = xlsread('UNet_ALL_data.xlsx');

%%  划分训练集和测试集
temp = randperm(length(res));
rate = 0.80;
P_train=res(temp(1: int16(length(res) * rate)), 1: 12)';
T_train=res(temp(1: int16(length(res) * rate)), 13)';
M = size(P_train, 2);
train_data = [P_train', T_train']; % 将P_train和T_train拼接成一个矩阵
train_data_filename = fullfile(outputFolder, 'train_data.xlsx');
xlswrite(train_data_filename, train_data, 1, 'A1'); 

P_test=res(temp(int16(length(res) * rate) + 1: end), 1: 12)';
T_test=res(temp(int16(length(res) * rate) + 1: end), 13)';
N = size(P_test, 2);
test_data = [P_test', T_test']; % 将P_train和T_train拼接成一个矩阵
test_data_filename = fullfile(outputFolder, 'test_data.xlsx');
xlswrite(test_data_filename, test_data, 1, 'A1');
%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

tongyimodelFilename = fullfile(outputFolder, 'ps_input.mat');
save(tongyimodelFilename, 'ps_input');
%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  创建模型
c = 15.0;      % 惩罚因子
g = 0.06;      % 径向基函数参数
cmd = ['-t 2', '-c', num2str(c), '-g', num2str(g)];
model = svmtrain(t_train, p_train, cmd);

%%  仿真测试
T_sim1 = svmpredict(t_train, p_train, model);
T_sim2 = svmpredict(t_test , p_test , model);

%%  性能评价
error1 = sum((T_sim1' == T_train)) / M * 100;
error2 = sum((T_sim2' == T_test )) / N * 100;

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  保存模型
modelFilename = fullfile(outputFolder, 'svm_model.mat');
save(modelFilename, 'model');


% %%  绘图
% figure
% plot(1: M, T_train, 'b-O', 1: M, T_sim1, 'r-*', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
% title(string)
% grid
% saveas(gcf, fullfile(outputFolder, 'Train_Prediction_Comparison.png'));
% % close;
% 
% figure
% plot(1: N, T_test, 'b-O', 1: N, T_sim2, 'r-x', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
% title(string)
% grid
% saveas(gcf, fullfile(outputFolder, 'Test_Prediction_Comparison.png'));
% % close;
  
%% 绘图
figure
scatter(1:M, T_train, 'b', 'O');  % 绘制训练集真实值散点图
hold on
scatter(1:M, T_sim1, 'r', '*');        % 绘制训练集预测值散点图
hold off
legend('True Values', 'Predicted Values')
xlabel('Sample')
ylabel('Prediction Results')
string = {'Comparison of Training Set Predictions'; ['Accuracy=' num2str(error1) '%']};
title(string)
grid
% xticks(1:300:M);  % 设置 x 轴刻度，每隔 300 个样本显示一个刻度
% yticks([1 2 3]);  % 只显示 y 轴上的数字 1、2、3
saveas(gcf, fullfile(outputFolder, 'Train_Prediction_Comparison.fig'));
% saveas(gcf, fullfile(outputFolder, 'Train_Prediction_Comparison.jpg'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Train_Prediction_Comparison.png'))
print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Train_Prediction_Comparison.jpg'))

% 绘制测试集散点图
figure
scatter(1:N, T_test, 'b', 'O');  % 绘制测试集真实值散点图
hold on
scatter(1:N, T_sim2, 'r', '*');        % 绘制测试集预测值散点图
hold off
legend('True Values', 'Predicted Values')
xlabel('Sample')
ylabel('Prediction Results')
string = {'Comparison of Test Set Predictions'; ['Accuracy=' num2str(error2) '%']};
title(string)
grid
% xticks(0:300:M);  % 设置 x 轴刻度，每隔 300 个样本显示一个刻度
% yticks([1 2 3]);  % 只显示 y 轴上的数字 1、2、3
saveas(gcf, fullfile(outputFolder, 'Test_Prediction_Comparison.fig'));
% saveas(gcf, fullfile(outputFolder, 'Test_Prediction_Comparison.jpg'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Test_Prediction_Comparison.png'))
print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Test_Prediction_Comparison.jpg'))
%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Train.fig'));
% saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Train.jpg'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Confusion_Matrix_Train.png'))
print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Confusion_Matrix_Train.jpg'))
% close;
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Test.fig'));
% saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Train.jpg'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Confusion_Matrix_Test.png'))
print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Confusion_Matrix_Test.jpg'))
% close;