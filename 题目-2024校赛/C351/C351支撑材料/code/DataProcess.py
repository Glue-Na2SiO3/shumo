import pandas as pd
df1=pd.read_excel("meng.xlsx")
df2=pd.read_excel("总表kaixin(5)111.xlsx")
name=[]
name1=[]
standard=df1["Date/Time"]
for i in range(len(standard)):
    name.append(standard[i])
current=df2["Date/Time"]
for i in range(len(current)):
    temp=""
    j=0
    change1=""
    change2=""
    flag=1
    while j<len(current[i]):
       while current[i][j]!="/" and flag:
           temp+=current[i][j]
           j+=1
           flag=0
       if current[i][j]=="/" :
            j+=1
            while current[i][j]!="/":
                change1+=current[i][j]
                j+=1
            j+=1
            while current[i][j]!=" ":
                change2+=current[i][j]
                j+=1
            temp+="/"+change2+"/"+change1
       temp+=current[i][j]
       j+=1
    name1.append(temp)
data=[]
have=[]
without=[]
num=0
for i in name:
    flag=1
    for j in range(len(current)):
            if name1[j]==i:
                data.append(df2[j:j+1].values.tolist()[0])
                have.append(num)
                flag=0
    if flag:
        data.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        without.append(num)
    num+=1
print(data)
# df3=pd.DataFrame(data=data,columns=["Date/Time","Line Temperature [℃] DEV1","Line Temperature [℃] DEV2","Ambient Temperature OTLM [℃] DEV1","Ambient Temperature OTLM [℃] DEV2","Line Current [A] DEV1","Line Current [A] DEV2","Humidity [%] DEV1","	Humidity [°] DEV2","Angle [°] DEV1","Angle [°] DEV2","longitude","latitude","altitude [m]","Vbat [V]","ACC1 RMS [mm/s]","ACC1 PA1 [mm/s]","ACC1 PF1 [Hz]","ACC1 PA2 [mm/s]","ACC1 PF2 [Hz]","ACC1 PA3 [mm/s]","ACC1 PF3 [Hz]","ACC1 Fcross [Hz]","ACC2 RMS [mm/s]","ACC2 PA1 [mm/s]","ACC2 PF1 [Hz]","ACC2 PA2 [mm/s]","ACC2 PF2 [Hz]","ACC2 PA3 [mm/s]","ACC2 PF3 [Hz]","ACC2 Fcross [Hz]","Stress1 Ch1 Avg [MPa]","Stress1 Ch1 MaxAvg [MPa]",	"Stress1 Ch1 MinAvg [MPa]","Stress1 Ch1 StdDev [MPa]","Stress1 Ch2 Avg [MPa]","Stress1 Ch2 MaxAvg [MPa]","Stress1 Ch2 MinAvg [MPa]","Stress1 Ch2 StdDev [MPa]","Stress2 Ch1 Avg [MPa]","Stress2 Ch1 MaxAvg [MPa]","Stress2 Ch1 MinAvg [MPa]","Stress2 Ch1 StdDev [MPa]","Stress2 Ch2 Avg [MPa]","Stress2 Ch2 MaxAvg [MPa]","Stress2 Ch2 MinAvg [MPa]","Stress2 Ch2 StdDev [MPa]","Stress3 Ch1 Avg [MPa]","Stress3 Ch1 MaxAvg [MPa]","Stress3 Ch1 MinAvg [MPa]","Stress3 Ch1 StdDev [MPa]","Stress3 Ch2 Avg [MPa]","Stress3 Ch2 MaxAvg [MPa]","Stress3 Ch2 MinAvg [MPa]","Stress3 Ch2 StdDev [MPa]","Stress4 Ch1 Avg [MPa]","Stress4 Ch1 MaxAvg [MPa]","Stress4 Ch1 MinAvg [MPa]","Stress4 Ch1 StdDev [MPa]","Stress4 Ch2 Avg [MPa]","Stress4 Ch2 MaxAvg [MPa]","Stress4 Ch2 MinAvg [MPa]","Stress4 Ch2 StdDev [MPa]"])
# df3.to_excel("Result.xlsx")
# df4=pd.Series(have)
# df5=pd.Series(without)
# df4.to_excel("WeHave.xlsx")
# df5.to_excel("WeDoNotHave.xlsx")