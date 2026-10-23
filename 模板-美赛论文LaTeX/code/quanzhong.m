clc,clear
WeatherWeight=[1 1/4 1/3 1/6 2 1/2
               4 1 2 1/3 3 2
               3 1/2 1 1/4 2 1
               6 3 4 1 7 3
               1/2 1/3 1/2 1/7 1 1/3
               2 1/2 1 1/3 3 1];
[n,n] = size(WeatherWeight);
Sum_WeatherWeight= sum(WeatherWeight);
SUM_WeatherWeight = repmat(Sum_WeatherWeight,n,1);
Stand_WeatherWeight = WeatherWeight ./ SUM_WeatherWeight;
disp('Result1:');
disp(sum(Stand_WeatherWeight,2)./n)


