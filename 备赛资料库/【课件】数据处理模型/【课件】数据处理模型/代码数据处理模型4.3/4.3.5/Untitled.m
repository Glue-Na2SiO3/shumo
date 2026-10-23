%AMCM86A
data=[129.0,7.5,4;
    140.0,141.5,8;
    108.5,28.0,6;
    88.0,147.0,8;
    185.5,22.5,6;
    195.0,137.5,8;
    105.5,85.5,8;
    157.5,-6.5,9;
    107.5,-81.0,9;
    77.0,3.0,8;
    81.0,56.5,8;
    162.0,84.0,4;
    117.5,-38.5,9;
    162.0,-66.5,9];%水道测量数据
plot(data(:,1),data(:,2),'*')
xx=[75,-50;
    200,-50;
    200,150;
    75,150;
    75,-50];
hold on;
plot(xx(:,1),xx(:,2),'r')
xlabel('X');
ylabel('Y');
title('水道离散点平面图');
 pause
d=zeros(14,1);
w=zeros(14,1);

m=50;  %x剖分区间数
n=80;  %y剖分区间
dx=(200-75)/m;%x剖分间隔
dy=(150+50)/n; %y剖分间隔
point=(m+1)*(n+1);%总点数
T=zeros(point,3);%存储各点数据
X=zeros(n+1,m+1);
Y=zeros(n+1,m+1);
Z=zeros(n+1,m+1);

p=3; 
  k=0;
  tp=0;
 for ix=1:m+1
     for iy=1:n+1
         x=75+(ix-1)*dx;   %x坐标
         y=-50+(iy-1)*dy; %y坐标
       for i=1:14
            %(x,y)格点到各已知各点距离
           d(i)=sqrt((x-data(i,1))^2+(y-data(i,2))^2);             
           w(i)=1.0/d(i)^p;%各点权重     
       end
       s=sum(w);
        w=w/s;%权值归一化
        z=sum(data(:,3).*w);     
         %求得网格点
        X(iy,ix)=x; Y(iy,ix)=y; Z(iy,ix)=-z;%Z坐标取负值便于绘制海底曲面图
        k=k+1;
         %存储各点(x,y,z)坐标
        T(k,1)=x; T(k,2)=y; T(k,3)=z;
         if z<=5
         tp=tp+1;
         D(tp,1)=x; D(tp,2)=y;%获得水深低于5英尺的点
      end
     end
 end


subplot(2,1,1);
   surf(X,Y,Z) %作曲面图
   xlabel('X');
   ylabel('Y');
   zlabel('Z');
   title('水道测量水底形状图')
subplot(2,1,2);
   contour(X,Y,Z); %作等值线图
   hold on
   plot(D(:,1),D(:,2),'*')%做出水深低于5英尺的点
   xlabel('X');
   ylabel('Y');
   title('水道测量等值线图');
   grid on
   hold off
