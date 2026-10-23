Commend=input('Commend=');
[n,m]=size(Commend);
Weight=[0.064066667 0.2035 0.122333333 0.431433333 0.051266667 0.1274 1];
% % % % % % % % % % % % %1:Standardization% % % % % % % % % % % % %
temp=zeros(1,m);
for i=1:m
    for j=1:n
        temp(1,i)=temp(1,i)+Commend(j,i).^2;
    end
end
StandardizedCommend=zeros(n,m);
for i=1:n
    for j=1:m
        StandardizedCommend(i,j)=Commend(i,j)/temp(1,j).^0.5;
    end
end
% % % % % % % % % % % % %2:Distance Calculation% % % % % % % % % % % % %
StandardizedCommend_Max=max(StandardizedCommend);
StandardizedCommend_Min=min(StandardizedCommend);
Distance_Max=zeros(n,1);
Distance_Min=zeros(n,1);
for i=1:n
    for j=1:m
        Distance_Max(i,1)=Distance_Max(i,1)+Weight(1,j).*(StandardizedCommend_Max(1,j)-StandardizedCommend(i,j)).^2;
        Distance_Min(i,1)=Distance_Min(i,1)+Weight(1,j).*(StandardizedCommend_Min(1,j)-StandardizedCommend(i,j)).^2;
    end
end
for i=1:n
     Distance_Max(i,1)=Distance_Max(i,1).^0.5;
     Distance_Min(i,1)=Distance_Min(i,1).^0.5;
end
% % % % % % % % % % % % %3:Score Calculation% % % % % % % % % % % % %
Score=zeros(n,1);
for i=1:n
    Score(i,1)=Distance_Max(i,1)/(Distance_Max(i,1)+Distance_Min(i,1));
end
% % % % % % % % % % % % %4:Normalization% % % % % % % % % % % % %
Score_Normalization=zeros(n,1);
Score_Sum=sum(Score);
for i=1:n    
    Score_Normalization(i,1)=Score(i,1)/Score_Sum;
end

