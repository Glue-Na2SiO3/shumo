%第三问做三次函数拟合
clear;

Y=xlsread('2016C附件','附件2');
[m,n]=size(Y);
S=[20,20,20,20];  %开始点序号
P=[301,301,301,148];  %结束点序号

for k=1:4  %状态序号1,2,3,4
 len=P(1)-S(1)+1;
 B=zeros(len,1);
 A=zeros(len,4);
 for i=S(1):P(1)
  p=i-S(1)+1; %序号   
 t=Y(i,k+1);  %时间
 U=Y(i,1);   %电压
 B(p)=t;
 A(p,1)=1; A(p,2)=U; A(p,3)=U*U;  A(p,4)=U*U*U;
 end
 
 [b,bint,r,rint,stats]=regress(B,A);
 
 NV=Y(S(1):P(1));
 NT=b(1)+b(2)*NV+b(3)*NV.*NV+b(4)*NV.*NV.*NV;
 plot(Y(S(k):P(k),1),Y(S(k):P(k),k+1),'b',NV,NT,'r');
 
 U=9.0;
 TE=b(1)+b(2)*U+b(3)*U*U+b(4)*U*U*U;
 TO=Y(P(k),k+1);
 dt=abs(TE-TO);  %绝对误差
 yt=dt/TO*100;  %相对误差
 if k<=3  fprintf('第k=%2d种状态，U=%2.1f,观测时间=%6.2f,估计时间%6.2f,绝对误差%3.2f，相对误差%6.2f%%\n',k,U,TO,TE,dt,yt);  end
 if k==4  fprintf('第k=%2d种状态，U=%2.1f,当前时间=%6.2f,衰减到9.0V估价时间%3.2f,剩余放电时间%6.2f分钟\r\n',k,U,TO,TE,dt);  end 
end
