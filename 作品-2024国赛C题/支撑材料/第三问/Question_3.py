import numpy as np
import random

field=np.zeros((20,26))

square=[80,55,35,72,68,55,60,46,40,28,25,86,55,44,50,25,60,45,35,20,15,13,15,18,27,20]

yield_now=[400,500,400,350,415,800,1000,400,630,525,110,3000,2200,420,525]

cost=[420,420,367.5,367.5,367.5,472.5,525,378,420,378,367.5,1050,2100,420,367.5]

price=[3.25,7.5,8.25,7,6.75,
       3.5,3,6.75,6,7.5,
       40,1.5,3.25,5.5,3.5]

plan_2023=[6,7,7,1,4,8,6,2,3,4,5,8,6,8,9,10,1,7,14,15,11,12,1,13,6,3]

sale=[17841.41088,17841.41088,18833.86961,18833.86961,18833.86961,16966.00025,16182.91875,18624.49238,17841.41088,18624.49238,18833.86961,11031.17165,14999.93092,17841.41088,18833.86961]

#初始化种群
for j in range(20):
    for i in range(26):
        field[j][i]=random.randint(1,15)
        while field[j][i]==plan_2023[i] and square[i]*yield_now[int(field[j][i])-1]>sale[int(field[j][i])-1]:
            field[j][i] = random.randint(1, 15)

Gtime=200
q_poss=0.7
p_poss=0.1
fitness=[]
fitBest=0
fBest=[0,0,0,0,0,
       0,0,0,0,0,
       0,0,0,0,0,
       0,0,0,0,0,
       0,0,0,0,0,
       0]
for i in range(20):
    sum=0
    for j in range(26):
        sum+=square[j]*yield_now[int(field[i][j])-1]*0.9*price[int(field[i][j])-1]-square[j]*cost[int(field[i][j])-1]*1.05
    fitness.append(sum)
for i in range(Gtime):
    #选择最优个体
    maxfit=max(fitness)
    minfit=min(fitness)
    for k in range(20):
        if fitness[k]==maxfit:
            maxindex=k
        if fitness[k]==minfit:
            minindex=k
    if maxfit>fitBest:
        fitBest=maxfit
        for m in range(26):
            fBest[m]=field[maxindex][m]

    #轮盘赌法选择遗传至下一代的个体
    fitness_sum=0
    pan=[]
    for q in range(20):
        fitness_sum+=fitness[q]
    fitness_chose=[0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0]
    fitness_cumsum = [0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0]
    for q in range(20):
        fitness_chose[q]=fitness[q]/fitness_sum #计算适应度累积百分比
    for q in range(20):
        for m in range(q+1):
            fitness_cumsum[q]+=fitness_chose[m]
    for q in range(20):
        pan.append(random.random())
    pan.sort()
    #print(pan)
    for q in range(20):
        if fitness_chose[q]<pan[q]:
            for p in range(26):
                field[q][p] = random.randint(1, 15)
                while field[q][p] == plan_2023[p] and square[p] * yield_now[int(field[q][p]) - 1] > sale[int(field[q][p]) - 1]:
                    field[q][p] = random.randint(1, 15)
    #模拟生物进化染色体交叉
    for m in range (0,20,2):
        for t in range(26):
            q=random.random()
            if q<q_poss:
                temp=field[m][t]
                field[m][t]=field[m+1][t]
                field[m + 1][t]=temp
    #模拟生物进化基因突变
    for m in range(20):
        for t in range(26):
            p=random.random()
            if p<p_poss:
                field[m][t]=random.randint(1,15)
                while field[m][t]==plan_2023[t]:
                    field[m][t] = random.randint(1, 15)
    for x in range(20):
        sum = 0
        for y in range(26):
            sum += square[y] * yield_now[int(field[x][y]) - 1] * price[int(field[x][y]) - 1] - square[y] * cost[int(field[x][y]) - 1] * price[int(field[x][y]) - 1]
        fitness[x]=sum

print(fitBest)
print(fBest)








