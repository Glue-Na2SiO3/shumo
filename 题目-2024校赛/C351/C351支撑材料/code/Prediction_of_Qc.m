clear,clc;
load matlab111.mat
%%导线直径为D
D=0.0096;
%%一般风速为v
v=0.6;
ta=PA;
tc=PL;
%%导体表面空气膜的热传导率为lambda
lambda=2.42*10^(-2)+7*10^(-5).*(0.5*ta+0.5*tc);
%%雷诺数为Re
Re=1.644*10^(9)*v*D.*(0.5*tc+0.5*ta).^(-1.78);
%%欧拉数为Nu
Nu=0.65.*Re.^(0.2)+0.23.*Re.^(0.61);
%%最终的对流散热功率为Qc
Qc=lambda.*Nu.*pi.*(tc-ta);