function X=BoatNumber
%产生一个到达船数的随机数  
  Boat=0:7; %到达船数取值范围
  %到达船数概率分布
  Prob=[0.05,0.1,0.1,0.25,0.20,0.15,0.1,0.05];
  n=length(Prob);
  Qu=zeros(1,n+1);
  Qu(1)=0;
  for i=1:n
    Qu(i+1)=Qu(i)+Prob(i); %产生概率区间
  end  
   Qu(n+1)=1.01; %将最后一个数值超过1,便于后面的随机数r取到1
          
  %产生一次到达船数  
  r=rand(1); %产生一个[0,1]随机变量  
  for i=1:n
   if(r>=Qu(i)&&r<Qu(i+1)) X=Boat(i); %获得到达船数
   end   
  end    
return  
