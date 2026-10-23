%问题1拟合得到4次曲线并计算MRE(相对误差估计精度)
clear;
X=xlsread('2016C附件','附件1');

[m,n]=size(X);
S=[20,20,20,20,20,20,20,20,40,30];  %开始点序号
N=[1883,1883,1228,863,655,523,432,366,311,270];  %各列数据总数
t98=[0,2826,1860,1294, 980, 766,606,498,412,344]; %V=9.8的最近观测时间
t90=[0,3764,2454,1724,1308,1044,862,730,620,536]; %V=9.0的最近观测时间


 %得到4次拟合曲线
 
 k=2;  %列序号

 for k=2:10
  
     t=zeros(N(k)-S(k)+1,1);
     v=zeros(N(k)-S(k)+1,1);
 
 %1.多项式拟合    
 for i=S(k):N(k)
  t(i-S(k)+1)=X(i,1);
  v(i-S(k)+1)=X(i,k);
 end
 
 [P,S1] = polyfit(t,v,4);  %4次多项式拟合
  fprintf('电流强度%2d\n',k*10);
  fprintf('4次多项式拟合参数:\n');
 for i=1:5
  fprintf('%6.4e ',P(i));
 end
 fprintf('\n\n');

 nv=P(1)*t.^4+P(2)*t.^3+P(3)*t.^2+P(4)*t+P(5);

 plot(t,v,'b',t,nv,'r');
 
 %2.寻找UM=9.0开始的相邻不超过0.005V的231个点对应的(t,v)
 dv=abs(diff(v));
 kp=length(dv);
   st=zeros(kp+1,1);
 for i=kp:-1:S(k)
   if dv(i)<=0.005  st(i)=1; st(i+1)=1; end    
 end
 
    H=[]; %获得从UM=9.0开始的相邻不超过0.005V的231个点序号
    number=0;
 for i=kp+1:-1:S(k)
    if st(i)==1    H=[H,i];  number=number+1;end
    if number>=231 break;end
 end
 
   Total=min(sum(st),231);  %总共点数,如果不足231则取符合条件的点
   nt=zeros(Total,1);
   nv=zeros(Total,1);
 for i=1:Total
    m=H(i);
    nt(i)=X(m,1);  %对应231个点的时间与观测电压
    nv(i)=X(m,k);
 end
 
 %3 利用牛顿迭代法寻找v对应的时间t
     
 % fprintf('采用牛顿迭代法求根:\n');
 
    s=0; 
   for km=1:Total       
   U=nv(km);
  t0=nt(km);
 for i=1:5 %迭代次数控制
 ft=P(1)*t0^4+P(2)*t0^3+P(3)*t0^2+P(4)*t0+P(5)-U; %迭代函数
 dft=4*P(1)*t0^3+3*P(2)*t0^2+2*P(3)*t0+P(4);  %导数
 t1=t0-ft/dft;
 %fprintf('U=%6.4f,i=%2d,t0=%6.2f,t1=%6.2f\n',U,i,nt(km),t1);
 t0=t1;
 end
 ft1=t1;
 s=s+abs(t1-nt(km))/nt(km);
 %fprintf('U=%6.4f,t0=%6.2f,t1=%6.2f\n',U,nt(km),t1);
   end
 MRE=s/Total;
 fprintf('Total=%4d,MRE=%6.4f%%\n',Total,MRE*100);
 end
