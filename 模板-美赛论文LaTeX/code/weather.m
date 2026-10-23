clc,clear
WeatherWeight=[1 1/4 1/3 1/6 2 1/2
               4 1 2 1/3 3 2
               3 1/2 1 1/4 2 1
               6 3 4 1 7 3
               1/2 1/3 1/2 1/7 1 1/3
               2 1/2 1 1/3 3 1];
[n,n] = size(WeatherWeight);
[V,D] = eig(WeatherWeight);
Max_eig = max(max(D));
CI = (Max_eig - n) / (n-1);
RI=[0 0.0001 0.52 0.89 1.12 1.26 1.36 1.41 1.46 1.49 1.52 1.54 1.56 1.58 1.59];
CR=CI/RI(n);
disp('CI=');disp(CI);
disp('CR=');disp(CR);
if CR<0.10
    disp('Accepted');
else
    disp('To Modify');
end