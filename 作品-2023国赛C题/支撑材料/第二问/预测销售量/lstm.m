clc
clear
load matlab.mat
data=huacai;
data=data(:,1)';  %设置数据格式
dataTrain = data(1:1000);   %训练集
dataTest = data(1001:end);    %测试集

mu = mean(dataTrain);    
sig = std(dataTrain);    
dataTrainStandardized = (dataTrain - mu) / sig;    %标准化训练集数据
XTrain = dataTrainStandardized(1:end-1);     %设置预测步数为1
YTrain = dataTrainStandardized(2:end);  
numFeatures = 1;    %设置为单维单步时间序列分段
numResponses = 1;  
numHiddenUnits = 40;  %设置LSTM隐含层为200
 
layers = [ ...
    sequenceInputLayer(numFeatures)     %输入层   
    lstmLayer(numHiddenUnits)   %LSTM层
    fullyConnectedLayer(numResponses)   %全连接层  
    regressionLayer];     %计算均方误差
options = trainingOptions('adam', ...   %解释器
    'MaxEpochs',600, ...   %进行1000轮训练
    'GradientThreshold',1, ...  %设置梯度阈值
    'InitialLearnRate',0.15, ...      %设置初始学习率
    'LearnRateSchedule','piecewise', ...?%表示经过一定数量时期学习率就会乘一个系数以降低学习率
    'LearnRateDropPeriod',150, ...      
    'LearnRateDropFactor',0.08, ...      
    'Verbose',0,  ...  
    'Plots','training-progress');   
net = trainNetwork(XTrain,YTrain,layers,options); 
net = predictAndUpdateState(net,XTrain);  %初始化网络状态
[net,YPred] = predictAndUpdateState(net,YTrain(end));  %利用预测值更新网络状态
for i = 2:95
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu'); %利用前一个值预测后一个值，预测结果写入YPred
end

YPred = sig*YPred + mu;      %对预测值去标准化
rmse = sqrt(mean((YPred(1:85)-dataTest).^2)) ;   %预测值与真实值计算均方根误差 
disp(rmse);
subplot(2,1,1)
plot(dataTrain(1:end))   
hold on
idx = 1001:1095;  
plot(idx,YPred(1:95),'.-') 
hold on 
plot([1000,1001],[dataTrain(end),YPred(1)])
hold off
xlabel("Time")
ylabel("Case")
title("Forecast")
legend(["Observed" "Forecast"])
subplot(2,1,2)
plot(data)
xlabel("Time")
ylabel("Case")
title("Dataset")
figure(2)
idx = 1001:1095;  
plot(idx,YPred(1:95),'.-')  
hold off


