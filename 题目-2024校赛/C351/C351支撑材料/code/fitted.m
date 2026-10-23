predict=zeros(31,1)
for i=1:31
    predict(i,:)=sim(net1,398+i);
end
disp(predict)