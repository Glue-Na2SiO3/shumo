x=0:0.5:10;
y=0.2*exp(-0.2*x)+0.5*exp(-0.15*x).*sin(1.25*x);

plot(x,y)  %画原始数据图

net.trainParam.epochs=5000;  %设定迭代步数
net=newff([0,10],[6,1],{'tansig','tansig'}); %初始化网络

net=train(net,x,y);  %进行网络训练

x1=0:0.1:10;
y1=sim(net,x1);  %数据泛化

plot(x,y,'*',x1,y1,'r');  %作对比图
