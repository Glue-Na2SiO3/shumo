clc,clear
load matlab111.mat
ta=PA;
tc=PL;
%%集肤效应系数为k
k=0.0025
%%20℃下导线材料温度系数为alpha20
alpha20=0.00403
%%20℃下导线的电阻为R20
R20=0.0000000265.*300./((0.0096./2).^2.*pi)
%%tc温度下导线的直流电阻为Rd
Rd=R20.*(1+alpha20.*(tc-20))
%%tc温度下导线的交流电阻为Rd
Rc=(1+k).*Rd
