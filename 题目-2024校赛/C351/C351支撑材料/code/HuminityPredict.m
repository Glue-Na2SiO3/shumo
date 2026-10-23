netc=removedelay(net);
num_delays=34;
num_predict=31;
output_test=[61.87153082 58.70746621 64.55718222 61.70271434 59.5119592 55.31219168 73.572548 82.33565842 54.6140529 60.95388146 64.80972639 81.86145393 87.31392285 68.93546755 72.59378389 76.1763119 82.28985708 80.40620557 70.8708517 83.71372162 74.18389807 76.94271871 66.59096336 76.19015521 76.2527906 76.13432946 85.83264914 83.76076697 180.2688225 115.4139737 75.82341189 87.52985283 87.4091407 0];
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