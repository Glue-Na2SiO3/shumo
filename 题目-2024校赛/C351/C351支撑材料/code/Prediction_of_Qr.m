clear,clc;
load matlab111.mat
ta=PA;
tc=PL;
%%导线表面的辐射系数为epsilon
epsilon=(0.23+0.46)/2
%%导线直径为D
D=0.0096;
%%斯蒂芬-包尔兹曼常数为sigma
sigma=0.0000000567
%%最终的辐射散热功率为Qr
Qr=pi.*epsilon.*D.*sigma.*((tc+273).^4-(ta+273).^4)

