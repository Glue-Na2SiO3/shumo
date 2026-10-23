import pandas as pd
df=pd.read_excel("ResultPredict.xlsx")
factor1=[2.445053429,2.715453647,2.581248798,2.205454517,1.718477399,1.229490237,0.669433879,-0.116655639,-0.874612545,-1.41419421,-1.64268094,-1.68573702,-1.4982946,-1.34956198,-1.29807196,-1.30966410,-1.51032582,-1.65710806,-1.36849987,-0.828112744,-0.253273274,0.374222841,1.063758949,1.804199156]
factor2=[2.613649817,2.769558471,2.521677405,2.084803778,1.588996612,1.082707343,0.315580391,-0.413235973,-0.872054556,-1.15244619,-1.37557558,-1.55169089,-1.6989796,-1.78757227,-1.85809857,-1.91655444,-1.92754479,-1.65635647,-1.03350347,-0.477519947,0.170818315,0.902915448,1.527071613,2.143353537]
data1=[]
data2=[]
day1=df["LineTemperature"]
day2=df["AmbientTemperature"]
for i in range(len(day1)):
    data1.append(day1[i:i+1].values.tolist()[0])
for i in range(len(day2)):
    data2.append(day2[i:i+1].values.tolist()[0])
data_final1=[]
data_final2=[]
for i in range(len(data1)):
    for j in range(len(factor1)):
        data_final1.append(data1[i]*(1+factor1[j]*0.01))
for i in range(len(data2)):
    for j in range(len(factor2)):
        data_final2.append(data2[i]*(1+factor2[j]*0.01))

df1=pd.Series(data_final1)
df2=pd.Series(data_final2)
df1.to_excel("Line.xlsx")
df2.to_excel("Am.xlsx")