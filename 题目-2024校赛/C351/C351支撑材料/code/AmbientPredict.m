netc=removedelay(net);
num_delays=28;
num_predict=31;
output_test=[20.23876109 21.81081864 12.42175021 17.53889072 21.14575883 21.65904652 15.29935079 13.3387729 15.84095211 14.74634825 16.98509266 15.27299442 13.38098693 13.33946267 16.31398131 19.1605809 20.46084317 19.05545677 19.20210243 15.63294814 16.96782435 18.84091512 21.09409891 21.49265299 22.5915625 22.09207511 21.27275013 0];
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