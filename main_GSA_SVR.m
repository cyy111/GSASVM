tic % 计时器
%% 清空环境变量
close all
clear
clc
format compact
load('../毕设实验部分/20220103data/lng_413832605')
%E. Rashedi, H. Nezamabadi-pour and S. Saryazdi, 
%GSA: A Gravitational Search Algorithm，Information sciences, vol. 179，no. 13, pp. 2232-2248, 2009.
%https://blog.csdn.net/senlin16888/article/details/72476475
train_input(1,:)=lng_413832605(1:389);
train_input(2,:)=lng_413832605(2:390);
train_input(3,:)=lng_413832605(3:391);
train_output=[lng_413832605(4:392)]';
test_input(1,:)=lng_413832605(393:end-3);
test_input(2,:)=lng_413832605(394:end-2);
test_input(3,:)=lng_413832605(395:end-1);
test_output=[lng_413832605(396:end)]'

[input_train,rule1]=mapminmax(train_input);
[output_train,rule2]=mapminmax(train_output);
input_test=mapminmax('apply',test_input,rule1);
output_test=mapminmax('apply',test_output,rule2);
%% GSA优化参数
N=20; % 群体规模 Number of agents.
max_it=30; % 最大迭代次数 Maximum number of iterations (T).
ElitistCheck=1; % 如果ElitistCheck=1,则使用文献中的公式21；如果ElitistCheck=0，则用文献中的公式9.
Rpower=1;% 文献中公式7中的R的幂次数 power of 'R' in eq.7.
min_flag=1; % 取1求解极小值问题，取0求解极大值问题 1: minimization, 0: maximization.
objfun=@objfun_svm; % 目标函数
[Fbest,Lbest,BestChart,MeanChart]=GSA_svm(objfun,N,max_it,ElitistCheck,min_flag,Rpower,...
input_train,output_train,input_test,output_test);
% Fbest: 最优目标值 Best result.
% Lbest: 最优解 Best solution. The location of Fbest in search space.
% BestChart: 最优解变化趋势 The best so far Chart over iterations.
% MeanChart: 平均适应度函数值变化趋势 The average fitnesses Chart over iterations.
%% 打印参数选择结果
bestc=Lbest(1);
bestg=Lbest(2);
disp('打印选择结果');
str=sprintf('Best c = %g，Best g = %g',bestc,bestg);
disp(str)
%% 利用最佳的参数进行SVM网络训练
cmd_gwosvm = ['-s 3 -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwosvm = libsvmtrain(output_train,input_train,cmd_gwosvm);
%% SVM网络预测
[predict_label,accuracy,~] = libsvmpredict(output_test',input_test',model_gwosvm);
test_pre=mapminmax('reverse',predict_label',rule2);
test_pre = test_pre';
str1=sprintf('acc=%g',accuracy);
disp(str1);

err_pre=lng_413832605(396:end)-test_pre;
figure('Name','测试数据残差图')
set(gcf,'unit','centimeters','position',[0.5,5,30,5])
title('测试数据残差图','FontSize',12);
plot(err_pre,'*-');
figure('Name','原始-预测图')
plot(test_pre,'*r-');hold on;
plot(test_output','bo-');
legend('预测','原始');
title('预测-原始对比图','FontSize',12);
set(gcf,'unit','centimeters','position',[0.5,13,30,5])

result=[lng_413832605(396:end),test_pre]

MAE=mymae(lng_413832605(396:end),test_pre)
MSE=mymse(lng_413832605(396:end),test_pre)
MAPE=mymape(lng_413832605(396:end),test_pre)
%% 显示程序运行时间
toc
