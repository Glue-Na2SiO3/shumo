clear,clc
load Fit_Data.mat

% 对数据进行JB正态分布检验
for i=1:3
    [H(i),P(i)] = jbtest(Fit_Data(:,i),0.05);
    
end

% 输出正态分布检验结果，均为1，拒绝正态分布假设
disp(H)
disp(P)

%故选择Spearman相关系数
corr(Fit_Data,'type','Spearman')