clear,clc
load matlab111.mat
season=1 %%对于八月份数据，season取1
%%导线的直径为D
D=0.0096;
%%导线表面的吸热系数为alpha
alpha=(0.23+0.46)/2
%%直射点纬度为delta
delta=ForDelta(day)
%%当地维度为altitude
latitude=46.089
SolarAltitude=ForSolarAltitude(latitude,delta,season)
%%当前导线所处环境日照辐射强度为q
q=Forq(SolarAltitude)
Qs=alpha.*D.*q


function [q]=Forq(SolarAltitude)%%计算当前导线所处环境日照辐射强度q
    q=1098.*cos(SolarAltitude).*exp(-0.057./cos(SolarAltitude))
end


function [SolarAltitude]=ForSolarAltitude(latitude,delta,season)%%计算太阳高度角SolarAltitude
    if season==1
        SolarAltitude=90-abs(latitude-delta) %太阳直射点与当地纬度照射同一半球取减号
    else
        SolarAltitude=90-abs(latitude+delta) %太阳直射点与当地纬度照射不同一半球取加号
    end
end


function [delta]=ForDelta(day) %%计算直射点纬度delta
    delta=23.45.*sin(2.*pi.*(284+day)./365) %%day表示当前为当年的第day天
end
