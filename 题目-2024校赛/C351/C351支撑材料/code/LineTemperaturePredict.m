netc=removedelay(net);
num_delays=6;
num_predict=31;
output_test=[25.8 24.7 24.8 24.8 24.7 0];
predict=zeros(num_predict,1)
for i =1:num_predict
    targetSeriesl=tonndata(output_test,true,false);
    [xsl,xisl,aisl,tsl]=preparets(netc,{},{},targetSeriesl);
    yl=netc(xsl,xisl,aisl);
    predict(i,1)=cell2mat(yl(1));
    for j=1:num_delays-2
        output_test(1,j)=output_test(1,j+1);
    end 
    output_test(1,num_delays-1)=cell2mat(yl(1));
end
disp(predict)