n=10000; %模拟总天数
x=zeros(n,1); %存储每天到达船数
a=zeros(n,1); %存储每天需要卸货的船数
b=zeros(n,1); %存储每天实际卸货的船数
d=zeros(n,1); %存储每天推迟卸货的船数

for i=1:n
    x(i)=BoatNumber; %模拟n天到达船数
end

a(1)=x(1);
if a(1)>4 b(1)=4; %计算每天实际卸货船数
 else b(1)=a(1);
end    
d(1)=a(1)-b(1);


for i=2:n
    a(i)=x(i)+d(i-1); %计算每天需要卸货的船数
    if a(i)>4 b(i)=4; %计算每天实际卸货船数
     else b(i)=a(i);   end   
   d(i)=a(i)-b(i); 
 %计算每天推迟卸货的船数     
end

Total=sum(d); %计算总共推迟卸货船数
Aver=Total/n; %计算每天推迟卸货的平均船数
fprintf('每天推迟卸货的平均船数%6.2f\n',Aver);
