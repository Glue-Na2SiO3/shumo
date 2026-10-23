import numpy as np
import random
import pandas as pd

def init_field(*field):
    for i in range(len(square)):
        field[i][4] = square[i]
    for i in range(len(field_type)):
        if field_type[i] == "平旱地":
            field[i][5] = 1
        elif field_type[i] == "梯田":
            field[i][5] = 2
        elif field_type[i] == "山坡地":
            field[i][5] = 3
        elif field_type[i] == "水浇地":
            field[i][5] = 4
        elif field_type[i] == "智慧大棚":
            field[i][5] = 6
        else:
            field[i][5] = 5
    return True

def creat(*field):
    # 根据约束条件随机生成种植策略
    for i in range(26):  # 平旱地、梯田、山坡地
        if judge_bean(plan_2023[i][0],plan_2024[i][0]):
            field[i][0] = random.randint(1, 41)
            while judge(field[i][5], field[i][0], plan_2024[i][0], plan_2024[i][1], plan_2024[i][2], plan_2024[i][3],1) == False:
                field[i][0] = random.randint(1, 41)
        else:
            field[i][0]=random.choice([1,2,3,4,5])
        # field[i][0] = random.randint(1, 41)
        # while judge(field[i][5], field[i][0], plan_2023[i][0], plan_2023[i][1], plan_2023[i][2], plan_2023[i][3],1) == False:
        #     field[i][0] = random.randint(1, 41)
    for i in range(8):  # 水浇地
        if judge_bean(plan_2023[i+26][0],plan_2023[i+26][1]) or judge_bean(plan_2024[i+26][0],plan_2024[i+26][1]):
            field[i + 26][0] = random.randint(1, 41)
            while judge(field[i + 26][5], field[i + 26][0], plan_2024[i + 26][0], plan_2024[i + 26][1],plan_2024[i + 26][2], plan_2024[i + 26][3], 0) == False:
                field[i + 26][0] = random.randint(1, 41)
        else:
            field[i + 26][0] = random.choice([17,18,19])
        # field[i + 26][0] = random.randint(1, 41)
        # while judge(field[i + 26][5], field[i + 26][0], plan_2023[i + 26][0], plan_2023[i + 26][1],plan_2023[i + 26][2], plan_2023[i + 26][3], 0) == False:
        #     field[i + 26][0] = random.randint(1, 41)

        if field[i + 26][0] == 16:  # 为水稻单季种植
            continue
        else:
            for j in range(3):
                field[i + 26][j + 1] = random.randint(1, 41)
                season = 1
                if j >= 1:
                    season = 2
                while judge(field[i + 26][5], field[i + 26][j + 1], plan_2024[i + 26][0], plan_2024[i + 26][1],plan_2024[i + 26][2], plan_2024[i + 26][3], season) == False:
                    field[i + 26][j + 1] = random.randint(1, 41)

    for i in range(16):  # 普通大棚
        for j in range(4):
            season = 1
            if j >= 2:
                season = 2
            if judge_bean(plan_2023[i + 26 + 8][0],plan_2023[i + 26 + 8][1]) or judge_bean(plan_2024[i + 26 + 8][0],plan_2024[i + 26 + 8][1]):
                field[i + 26 + 8][j] = random.randint(1, 41)
                while judge(field[i + 26 + 8][5], field[i + 26 + 8][j], plan_2024[i + 26 + 8][0], plan_2024[i + 26 + 8][1],plan_2024[i + 26 + 8][2], plan_2024[i + 26 + 8][3], season) == False:
                    field[i + 26 + 8][j] = random.randint(1, 41)
            else:
                field[i + 26 + 8][j]=random.choice([17,18,19])

            # field[i + 26 + 8][j] = random.randint(1, 41)
            # while judge(field[i + 26 + 8][5], field[i + 26 + 8][j], plan_2023[i + 26 + 8][0], plan_2023[i + 26 + 8][1],plan_2023[i + 26 + 8][2], plan_2023[i + 26 + 8][3], season) == False:
            #     field[i + 26 + 8][j] = random.randint(1, 41)

    for i in range(4):  # 智能大棚
        for j in range(4):
            season = 1
            if j >= 2:
                season = 2
            if judge_bean(plan_2023[i + 26 + 8 + 16][0],plan_2023[i + 26 + 8 + 16][1]) or judge_bean(plan_2023[i + 26 + 8 + 16][2],plan_2023[i + 26 + 8 + 16][3]) or judge_bean(plan_2024[i + 26 + 8 + 16][0],plan_2024[i + 26 + 8 + 16][1]) or judge_bean(plan_2024[i + 26 + 8 + 16][2],plan_2024[i + 26 + 8 + 16][3]) :
                field[i + 26 + 8 + 16][j] = random.randint(1, 41)
            else:
                field[i + 26 + 8 + 16][j] = random.choice([17, 18, 19])
            # field[i + 26 + 8 + 16][j] = random.randint(1, 41)
            if season == 1:
                while judge(field[i + 26 + 8 + 16][5], field[i + 26 + 8 + 16][j], plan_2024[i + 26 + 8 + 16][0],plan_2024[i + 26 + 8 + 16][1], plan_2024[i + 26 + 8 + 16][2], plan_2024[i + 26 + 8 + 16][3],season) == False:
                    field[i + 26 + 8 + 16][j] = random.randint(1, 41)
            else:
                while judge(field[i + 26 + 8 + 16][5], field[i + 26 + 8 + 16][j], plan_2024[i + 26 + 8 + 16][0],plan_2024[i + 26 + 8 + 16][1], plan_2024[i + 26 + 8 + 16][2], plan_2024[i + 26 + 8 + 16][3],season) == False or judge(field[i + 26 + 8 + 16][5], field[i + 26 + 8 + 16][j],field[i + 26 + 8 + 16][0], field[i + 26 + 8 + 16][1],plan_2023[i + 26 + 8 + 16][2], plan_2023[i + 26 + 8 + 16][3],season) == False:
                    field[i + 26 + 8 + 16][j] = random.randint(1, 41)
    return True

def cost(* field):
    #print(cost_2023)
    cost_sum=0
    for i in range(54):
        x=int(field[i][0])
        y=int(field[i][5]-1)
        if field[i][1]==0: # 单季产品
            cost_sum+=cost_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]
        else:
            cost_sum+=cost_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*0.5+cost_2023[int(field[i][1]-1)][int(field[i][5]-1)]*field[i][4]*0.5+cost_2023[int(field[i][2]-1)][int(field[i][5]-1)]*field[i][4]*0.5+cost_2023[int(field[i][3]-1)][int(field[i][5]-1)]*field[i][4]*0.5
    return cost_sum

def profit_cal(year,target,* field):
    profit_sum=0
    if target==1:
        for i in range(54):
            x=int(field[i][0]-1)
            y=int(field[i][5]-1)
            if x<=16:
                if x==6 or x==7:
                    if field[i][1] == 0:
                        profit_sum += min(yield_2023[int(field[i][0] - 1)][int(field[i][5] - 1)] * field[i][4] * 0.9 ** (year),yield_2023[int(field[i][0] - 1)][int(field[i][5] - 1)] * field[i][4] * 1.05**(year)) * (price_2023[int(field[i][0] - 1)][int(field[i][5] - 1)][0] +price_2023[int(field[i][0] - 1)][int(field[i][5] - 1)][1]) / 2
                    else:
                        for k in range(4):
                            profit_sum += min(yield_2023[int(field[i][k] - 1)][int(field[i][5] - 1)] * field[i][4] * 0.9 ** (year),yield_2023[int(field[i][k] - 1)][int(field[i][5] - 1)] * field[i][4] *1.05**(year)) * (price_2023[int(field[i][k] - 1)][int(field[i][5] - 1)][0] +price_2023[int(field[i][k] - 1)][int(field[i][5] - 1)][1]) / 2 * 0.5

                else:
                    if field[i][1]==0:
                        profit_sum+=min(yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*1.05)*(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])/2
                    else:
                        for k in range(4):
                            profit_sum+=min(yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]*1.05)*(price_2023[int(field[i][k]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][k]-1)][int(field[i][5]-1)][1])/2*0.5
            elif x<=37:
                if field[i][1]==0:
                    profit_sum+=min(yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)*1.05)*(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])/2*1.05**(year)
                else:
                    for k in range(4):
                        profit_sum+=min(yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)*1.05)*(price_2023[int(field[i][k]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][k]-1)][int(field[i][5]-1)][1])/2*0.5*1.05**(year)
            elif x<41:
                if field[i][1] == 0:
                    profit_sum += min(yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)*1.05)* (price_2023[int(field[i][0] - 1)][int(field[i][5] - 1)][0] + price_2023[int(field[i][0] - 1)][int(field[i][5] - 1)][1]) / 2 * 0.99 ** ( year)
                else:
                    for k in range(4):
                        profit_sum +=min(yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)*1.05)* (price_2023[int(field[i][k] - 1)][int(field[i][5] - 1)][0] +price_2023[int(field[i][k] - 1)][int(field[i][5] - 1)][1]) / 2 * 0.5 *0.99 ** (year)
            else:
                if field[i][1] == 0:
                    profit_sum +=min(yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)*1.05)* (price_2023[int(field[i][0] - 1)][int(field[i][5] - 1)][0] + price_2023[int(field[i][0] - 1)][int(field[i][5] - 1)][1]) / 2 * 0.95 ** ( year)
                else:
                    for k in range(4):
                        profit_sum +=min(yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]* 0.9**(year),yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)*1.05)* (price_2023[int(field[i][k] - 1)][int(field[i][5] - 1)][0] +price_2023[int(field[i][k] - 1)][int(field[i][5] - 1)][1]) / 2 * 0.5 *0.95 ** (year)

        return profit_sum

#if sale_2023[int(field[i][0]-1)][int(field[i][5]-1)]<yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]:
#profit_sum+=sale_2023[int(field[i][0]-1)][int(field[i][5]-1)]*random.uniform(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0],price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])
#else:
#profit_sum+=yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*random.uniform(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0],price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])

    else:
        for i in range(54):
            x=int(field[i][0]-1)
            y=int(field[i][5]-1)
            if x<=16:
                if field[i][1]==0:
                    profit_sum+=yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])/2
                else:
                    for k in range(4):
                        profit_sum+=yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]*(price_2023[int(field[i][k]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][k]-1)][int(field[i][5]-1)][1])/2*0.5
            else:
                if field[i][1]==0:
                    profit_sum+=(yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)+yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*0.0108*0.5)*(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])/2
                else:
                    for k in range(4):
                        profit_sum+=(yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]*(1-0.0108)+yield_2023[int(field[i][k]-1)][int(field[i][5]-1)]*field[i][4]*0.0108*0.5)*(price_2023[int(field[i][k]-1)][int(field[i][5]-1)][0]+price_2023[int(field[i][k]-1)][int(field[i][5]-1)][1])/2*0.5
        return profit_sum

#if sale_2023[int(field[i][0]-1)][int(field[i][5]-1)]<yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]:
#profit_sum+=sale_2023[int(field[i][0]-1)][int(field[i][5]-1)]*random.uniform(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0],price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])+(yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]-sale_2023[int(field[i][0]-1)][int(field[i][5]-1)])*0.5*random.uniform(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0],price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])
#else:
#profit_sum+=yield_2023[int(field[i][0]-1)][int(field[i][5]-1)]*field[i][4]*random.uniform(price_2023[int(field[i][0]-1)][int(field[i][5]-1)][0],price_2023[int(field[i][0]-1)][int(field[i][5]-1)][1])



def target_1(year,*field):
    result=profit_cal(year,1,*field)-cost(*field)*1.05**(year)
    return result

def target_2(year,*field):
    result = profit_cal(year,2,*field) - cost(*field)*1.05**(year)
    return result

def judge(field_input,crop_input,crop_pre_1,crop_pre_2,crop_pre_3,crop_pre_4,crop_season): #判断该地块是否可以种植该作物
    enable_123=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    enable_4_1=[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    enable_4_2=[35,36,37]
    enable_4_special=16
    enable_5_1=[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    enable_5_2=[38,39,40,41]
    enable_6=[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    #判断平旱地、梯田、山坡地种植情况
    if field_input in [1,2,3]:
        if crop_input in enable_123 and crop_input != crop_pre_1:
            return True
        else:
            return False
    elif field_input == 4:
        if crop_season==0 and crop_input==enable_4_special:
            if crop_input != crop_pre_1:
                return True
            else:
                return False
        else:
            if crop_season==1 or crop_season==0:
                if crop_input in enable_4_1 and ((crop_input !=crop_pre_1 and crop_input !=crop_pre_2) and (crop_input !=crop_pre_3 and crop_input !=crop_pre_4)):
                    return True
                else:
                    return False
            else:
                if crop_input in enable_4_2 and ((crop_input !=crop_pre_1 and crop_input !=crop_pre_2) and (crop_input !=crop_pre_3 and crop_input !=crop_pre_4)):
                    return True
                else:
                    return False
    elif field_input == 5:
        if crop_season == 1:
            if crop_input in enable_5_1 and ((crop_input != crop_pre_1 and crop_input != crop_pre_2) and (crop_input != crop_pre_3 and crop_input != crop_pre_4)):
                return True
            else:
                return False
        else:
            if crop_input in enable_5_2 and ((crop_input != crop_pre_1 and crop_input != crop_pre_2) and (crop_input != crop_pre_3 and crop_input != crop_pre_4)):
                return True
            else:
                return False
    elif field_input == 6: #注意使用field_input==6时，crop_pre_1和crop_pre_2需要替换为上一季度的值再进行一次judge
        if crop_season == 1:
            if crop_input in enable_6 and ((crop_input != crop_pre_1 and crop_input != crop_pre_2) and (crop_input != crop_pre_3 and crop_input != crop_pre_4)):
                return True
            else:
                return False
        else:
            if crop_input in enable_6 and ((crop_input != crop_pre_1 and crop_input != crop_pre_2) and (crop_input != crop_pre_3 and crop_input != crop_pre_4)):
                return True
            else:
                return False
    else:
        return False


def judge_bean(crop_pre_1,crop_pre_2):
    bean_type=[1,2,3,4,5,17,18,19]
    if (crop_pre_1 not in bean_type) and (crop_pre_2 not in bean_type):
        return False
    else:
        return True


#创建地块矩阵 field
#简化为平旱地、梯田、山坡地仅种植一种，水浇地、普通大棚、智慧大棚最多平分种植两种作物
file1=pd.read_excel(r"D:\STUDY\数模国赛2024\C题\附件1.xlsx",sheet_name="乡村的现有耕地")
#print(file1)
square=file1["地块面积/亩"]
field_type=file1["地块类型"]
#print(field_type)
field=np.zeros((54,6)) #第一季度作物上半（单季），第一季度作物下半，第二季度作物上半，第二季度作物下半，地块面积，地块类型 1.平旱地 2.梯田 3.山坡地 4.水浇地 5.普通大棚 6.智慧大棚
field1=np.zeros((54,6))
field2=np.zeros((54,6))
field3=np.zeros((54,6))
field4=np.zeros((54,6))
field5=np.zeros((54,6))
fBest=np.zeros((54,6)) #用于存储最优解

init_field(*field)
init_field(*field1)
init_field(*field2)
init_field(*field3)
init_field(*field4)
init_field(*field5)

#print(field)
#print(field1)

#创建作物矩阵 crop
file2=pd.read_excel(r"D:\STUDY\数模国赛2024\C题\附件1.xlsx",sheet_name="乡村种植的农作物")
crop=np.zeros((41,2)) # 作物编号,作物类型 1.粮食（豆类） 2.粮食 3.蔬菜（豆类） 4.蔬菜 5.食用菌

crop_number=file2["作物编号"]
#print(crop_number)
crop_kind=file2["作物类型"]
for i in range(41):
    crop[i][0]=crop_number[i]
for i in range(41):
    if crop_kind[i] == "粮食（豆类）":
        crop[i][1]=1
    elif crop_kind[i] == "粮食":
        crop[i][1]=2
    elif crop_kind[i] == "蔬菜（豆类）":
        crop[i][1]=3
    elif crop_kind[i] == "蔬菜":
        crop[i][1]=4
    else:
        crop[i][1]=5
#print(crop)
#result1_1
# plan_2029=[[ 1,0,0,0],[ 4, 0, 0, 0],[ 7, 0, 0, 0],[ 3, 0, 0, 0],[11, 0, 0, 0],[ 5, 0, 0, 0],
#            [ 2, 0, 0, 0],[ 8, 0, 0, 0],[ 3, 0, 0, 0],[14, 0, 0, 0],[ 3, 0, 0, 0],[ 1, 0, 0, 0],
#            [13, 0, 0, 0],[ 3, 0, 0, 0],[ 8, 0, 0, 0],[ 2, 0, 0, 0],[ 3, 0, 0, 0],[ 4, 0, 0, 0],
#            [ 4, 0, 0, 0],[ 7, 0, 0, 0],[ 5, 0, 0, 0],[ 6, 0, 0, 0],[13, 0, 0, 0],[12, 0, 0, 0],
#            [ 2, 0, 0, 0],[12, 0, 0, 0],[17,31,37,37],[17,23,37,37],[17,29,35,37],[29,32,35,35],
#            [23,32,36,35],[28,23,36,37],[34,23,36,37],[17,18,35,36],[29,34,41,40],[19,18,17,19],
#            [18,19,19,17],[18,18,17,19],[18,17,18,18],[19,17,17,18],[32,21,38,38],[18,19,19,19],
#            [18,17,17,19],[17,18,17,17],[22,26,38,40],[30,32,38,41],[18,17,17,17],[21,20,38,41],
#            [34,33,38,38],[18,27,38,38],[27,21,19,33],[17,17,19,27],[32,32,21,31],[34,31,33,28]]
#
#
# plan_2028=[[14,0,0,0],[10, 0, 0, 0],[ 4, 0, 0, 0],[12, 0, 0, 0],[14, 0, 0, 0],[ 9, 0, 0, 0],
#            [ 4, 0, 0, 0],[ 3, 0, 0, 0],[ 1, 0, 0, 0],[ 9, 0, 0, 0],[15, 0, 0, 0],[13, 0, 0, 0],
#            [ 6, 0, 0, 0],[12, 0, 0, 0],[14, 0, 0, 0],[12, 0, 0, 0],[ 6, 0, 0, 0],[13, 0, 0, 0],
#            [ 2, 0, 0, 0],[ 1, 0, 0, 0],[12, 0, 0, 0],[ 8, 0, 0, 0],[ 1, 0, 0, 0],[ 2, 0, 0, 0],
#            [12, 0, 0, 0],[ 2, 0, 0, 0],[18,29,36,36],[32,21,35,35],[19,19,36,36],[34,23,37,37],
#            [19,24,37,37],[18,20,35,35],[17,22,35,35],[32,23,23,34],[33,31,39,38],[25,25,41,41],
#            [23,23,41,40],[30,33,41,41],[27,32,41,41],[20,29,31,19],[23,23,40,40],[27,32,25,17],
#            [25,24,39,40],[23,22,38,41],[18,25,27,30],[24,31,40,40],[24,25,39,40],[28,22,22,24],
#            [28,27,40,40],[17,21,41,40],[30,26,22,17],[23,23,30,21],[23,17,34,33],[23,18,29,32]]
#
#
# plan_2027=[[ 9, 0, 0, 0],[13, 0, 0, 0],[ 7, 0, 0, 0],[11, 0, 0, 0],[ 2, 0, 0, 0],[14, 0, 0, 0],
#            [12, 0, 0, 0],[ 2, 0, 0, 0],[ 4, 0, 0, 0],[ 3, 0, 0, 0],[ 9, 0, 0, 0],[ 9, 0, 0, 0],
#            [ 3, 0, 0, 0],[ 4, 0, 0, 0],[ 2, 0, 0, 0],[10, 0, 0, 0],[ 4, 0, 0, 0],[ 8, 0, 0, 0],
#            [ 8, 0, 0, 0],[12, 0, 0, 0],[15, 0, 0, 0],[ 4, 0, 0, 0],[ 6, 0, 0, 0],[10, 0, 0, 0],
#            [ 9, 0, 0, 0],[10, 0, 0, 0],[22,25,35,37],[30,20,37,36],[17,22,37,35],[21,17,36,35],
#            [34,25,36,35],[32,23,37,37],[19,28,37,36],[27,24,23,29],[17,18,19,17],[34,29,39,38],
#            [20,28,39,39],[25,34,39,40],[24,25,39,38],[34,25,38,41],[18,19,19,18],[23,25,39,39],
#            [28,32,41,38],[28,27,39,40],[23,34,40,38],[17,17,19,19],[26,21,38,38],[18,19,17,17],
#            [18,31,38,39],[19,23,39,38],[19,32,33,23],[26,27,30,28],[18,30,20,25],[20,25,27,24]]
#
# plan_2026=[[ 6,0,0,0],[ 3,0,0,0],[13,0,0,0],[ 2,0,0,0],[11,0,0,0],[ 2,0,0,0],
#            [ 6,0,0,0],[ 1,0,0,0],[11,0,0,0],[ 2,0,0,0],[ 1,0,0,0],[ 3,0,0,0],
#            [15,0,0,0],[11,0,0,0],[13,0,0,0],[ 4,0,0,0],[ 3,0,0,0],[ 2,0,0,0],
#            [ 7,0,0,0],[ 9,0,0,0],[ 5,0,0,0],[12,0,0,0],[12,0,0,0],[ 9,0,0,0],
#            [ 4,0,0,0],[15,0,0,0],[31,8,5,7],[27,4,6,6],[32,9,7,7],[31,0,7,7],
#            [17,3,5,7],[17,9,7,6],[29,0,5,6],[18,5,6,6],[33,9,7,8],[19,8,0,9],
#            [24,4,9,2],[18,8,8,8],[17,5,9,1],[23,9,8,0],[27,8,0,3],[22,3,4,2],
#            [18,7,7,9],[17,7,9,9],[24,0,0,8],[25,7,9,1],[19,8,9,8],[32,0,9,9],
#            [19,7,8,9],[18,9,9,7],[17,7,8,2],[18,9,7,3],[29,8,0,5],[30,7,1,4]]
#
# plan_2025=[[5,0,0,0],[10, 0, 0, 0],[ 5, 0, 0, 0],[ 8, 0, 0, 0],[15, 0, 0, 0],[11, 0, 0, 0],
#            [ 1, 0, 0, 0],[12, 0, 0, 0],[ 7, 0, 0, 0],[ 8, 0, 0, 0],[ 7, 0, 0, 0],[11, 0, 0, 0],
#            [ 2, 0, 0, 0],[ 5, 0, 0, 0],[ 6, 0, 0, 0],[10, 0, 0, 0],[ 5, 0, 0, 0],[14, 0, 0, 0],
#            [ 4, 0, 0, 0],[ 1, 0, 0, 0],[15, 0, 0, 0],[10, 0, 0, 0],[ 2, 0, 0, 0],[ 2, 0, 0, 0],
#            [ 7, 0, 0, 0],[ 3, 0, 0, 0],[29,19,36,36],[33,19,37,35],[16,21,35,36],[19,32,35,35],
#            [27,26,36,36],[24,29,35,35],[27,23,37,37],[32,21,35,37],[26,32,41,41],[21,21,41,38],
#         [17,27,41,38],[21,32,39,38],[23,27,40,38],[18,34,39,41],[27,23,40,38],[27,29,40,39],
#         [32,21,41,41],[21,18,38,41],[18,30,41,41],[22,34,40,40],[24,21,39,39],[24,23,41,41],
#         [20,28,39,39],[32,29,39,41],[19,22,34,29],[25,26,22,22],[24,31,21,21],[27,27,26,25]]
#
#
# plan_2024=[[2,0,0,0],[15, 0, 0, 0],[ 8, 0, 0, 0],[ 9, 0, 0, 0],[10, 0, 0, 0],[14, 0, 0, 0],
#            [ 3, 0, 0, 0],[ 6, 0, 0, 0],[ 2, 0, 0, 0],[13, 0, 0, 0],[ 8, 0, 0, 0],[ 1, 0, 0, 0],
#            [ 1, 0, 0, 0],[ 2, 0, 0, 0],[ 2, 0, 0, 0],[ 6, 0, 0, 0],[ 4, 0, 0, 0],[11, 0, 0, 0],
#            [ 3, 0, 0, 0],[ 6, 0, 0, 0],[ 6, 0, 0, 0],[ 3, 0, 0, 0],[ 4, 0, 0, 0],[ 7, 0, 0, 0],
#            [12, 0, 0, 0],[12, 0, 0, 0],[18,22,37,35],[25,23,36,36],[23,19,37,37],[29,26,37,36],
#            [25,21,37,35],[26,34,36,36],[30,17,36,36],[28,29,36,36],[33,31,39,40],[22,17,39,40],
#            [18,28,39,40],[22,30,41,40],[34,18,41,41],[25,31,40,38],[22,24,39,41],[34,24,38,41],
#            [23,27,39,39],[27,23,40,39],[33,22,38,40],[19,33,38,39],[28,34,38,40],[17,31,40,38],
#            [29,32,38,38],[28,27,38,38],[27,30,31,17],[33,23,34,30],[33,19,18,29],[29,31,18,20]]

#2023的种植情况，按照field土地顺序

#result1_2

#result 2
plan_2029=[[10,0,0,0],[ 2, 0, 0, 0],[ 8, 0, 0, 0],[ 6, 0, 0, 0],[ 9, 0, 0, 0],[ 2, 0, 0, 0],[11, 0, 0, 0],[ 7, 0, 0, 0],[ 5, 0, 0, 0],[ 5, 0, 0, 0],[ 6, 0, 0, 0],[ 9, 0, 0, 0],[ 4, 0, 0, 0],[ 3, 0, 0, 0],[ 9, 0, 0, 0],[13, 0, 0, 0],[ 2, 0, 0, 0],[11, 0, 0, 0],[ 5, 0, 0, 0],[14, 0, 0, 0],[ 1, 0, 0, 0],[ 4, 0, 0, 0],[ 1, 0, 0, 0],[ 5, 0, 0, 0],[ 2, 0, 0, 0],[ 2, 0, 0, 0],[24,32,36,36],[28,33,35,35],[18,29,37,37],[23,25,37,37],[22,17,36,36],[29,22,36,36],[18,17,36,36],[29,29,36,36],[17,18,19,17],[22,32,18,27],[17,19,17,17],[18,24,40,39],[29,28,41,39],[23,21,41,39],[26,21,39,38],[19,17,17,17],[17,17,19,19],[32,25,41,39],[19,17,19,19],[17,19,18,18],[28,31,40,41],[19,33,41,41],[32,23,40,40],[26,32,39,39],[17,19,33,21],[27,31,29,21],[31,27,25,20],[30,20,26,34]]

plan_2028=[[ 3,0,0,0],[12, 0,  0, 0],[ 2, 0,  0, 0],[13, 0,  0, 0],[12, 0,  0, 0],[13, 0,  0, 0],[ 2, 0,  0, 0],[ 5, 0,  0, 0],[12, 0,  0, 0],[14, 0,  0, 0],[ 1, 0,  0, 0],[13, 0,  0, 0],[ 3, 0,  0, 0],[14, 0,  0, 0],[ 4, 0,  0, 0],[ 4, 0,  0, 0],[13, 0,  0, 0],[ 3, 0,  0, 0],[ 8, 0,  0, 0],[ 5, 0,  0, 0],[ 6, 0,  0, 0],[ 2, 0,  0, 0],[ 8, 0,  0, 0],[12, 0,  0, 0],[12, 0,  0, 0],[11, 0,  0, 0],[19,17, 35,35],[17,23, 37,36],[24,26, 27,28],[18,28, 36,35],[33,33, 37,35],[17,33, 37,37],[29,24, 35,37],[17,28, 37,35],[22,21, 32,32],[23,18, 28,34],[28,27, 17,18],[25,21, 33,32],[18,18, 19,17],[19,19, 17,18],[17,19, 19,17],[33,23, 40,40],[23,34, 34,26],[19,17, 18,19],[24,33, 40,40],[33,25, 34,31],[33,33, 38,38],[22,28, 38,40],[18,19, 17,18],[19,19, 18,18],[27,32, 22,22],[34,18, 22,33],[18,19, 17,24],[32,27, 24,17]]


plan_2027=[[13,0,0,0],[10, 0, 0, 0],[ 5, 0, 0, 0],[ 2, 0, 0, 0],[ 5, 0, 0, 0],[ 2, 0, 0, 0],[ 9, 0, 0, 0],[10, 0, 0, 0],[ 7, 0, 0, 0],[ 6, 0, 0, 0],[12, 0, 0, 0],[ 3, 0, 0, 0],[15, 0, 0, 0],[ 5, 0, 0, 0],[ 8, 0, 0, 0],[ 7, 0, 0, 0],[ 9, 0, 0, 0],[ 9, 0, 0, 0],[15, 0, 0, 0],[10, 0, 0, 0],[ 4, 0, 0, 0],[10, 0, 0, 0],[ 7, 0, 0, 0],[ 4, 0, 0, 0],[12, 0, 0, 0],[ 9, 0, 0, 0],[32,34,36,36],[31,31,35,35],[29,23,36,36],[29,34,37,37],[29,18,27,21],[26,26,32,19],[32,30,36,36],[23,34,36,36],[25,27,40,38],[18,34,39,38],[33,24,40,41],[19,33,39,38],[31,29,34,22],[29,29,21,23],[33,33,25,30],[25,29,41,38],[20,24,39,39],[21,24,28,26],[32,21,38,39],[20,30,24,29],[19,17,19,18],[19,17,17,18],[32,29,41,41],[32,31,39,40],[29,30,23,20],[29,17,24,19],[34,23,23,31],[26,28,31,33]]

plan_2026=[[ 9,0,0,0],[13, 0, 0, 0],[ 7, 0, 0, 0],[11, 0, 0, 0],[ 7, 0, 0, 0],[10, 0, 0, 0],[10, 0, 0, 0],[ 4, 0, 0, 0],[ 5, 0, 0, 0],[ 3, 0, 0, 0],[ 7, 0, 0, 0],[11, 0, 0, 0],[ 7, 0, 0, 0],[ 9, 0, 0, 0],[14, 0, 0, 0],[ 2, 0, 0, 0],[ 3, 0, 0, 0],[ 4, 0, 0, 0],[ 3, 0, 0, 0],[ 9, 0, 0, 0],[14, 0, 0, 0],[13, 0, 0, 0],[ 2, 0, 0, 0],[13, 0, 0, 0],[12, 0, 0, 0],[ 4, 0, 0, 0],[29,31,37,35],[32,22,37,36],[31,31,30,33],[28,32,35,35],[32,29,37,36],[16,33,36,35],[29,19,35,37],[27,29,25,19],[18,17,19,17],[26,30,22,31],[20,29,28,34],[17,28,40,40],[24,28,38,40],[34,31,39,39],[31,29,39,40],[19,17,18,17],[21,17,41,40],[25,27,24,23],[19,18,17,19],[21,17,38,40],[25,26,40,39],[34,27,24,18],[24,23,40,40],[26,30,41,38],[18,22,17,18],[26,30,23,20],[24,24,23,25],[17,17,25,29]]

plan_2025=[[ 4,0,0,0],[ 2, 0, 0, 0],[12, 0, 0, 0],[ 7, 0, 0, 0],[11, 0, 0, 0],[ 3, 0, 0, 0],[ 4, 0, 0, 0],[ 6, 0, 0, 0],[12, 0, 0, 0],[ 7, 0, 0, 0],[ 1, 0, 0, 0],[ 7, 0, 0, 0],[ 3, 0, 0, 0],[14, 0, 0, 0],[ 4, 0, 0, 0],[ 1, 0, 0, 0],[ 9, 0, 0, 0],[ 1, 0, 0, 0],[ 5, 0, 0, 0],[ 4, 0, 0, 0],[11, 0, 0, 0],[ 2, 0, 0, 0],[10, 0, 0, 0],[ 1, 0, 0, 0],[ 3, 0, 0, 0],[14, 0, 0, 0],[17,33,36,36],[19,21,35,35],[17,17,35,35],[18,18,37,37],[19,17,35,35],[17,32,37,37],[18,34,36,36],[19,31,35,35],[21,34,38,40],[17,18,18,19],[17,19,17,18],[20,20,41,41],[17,18,18,19],[25,26,40,40],[18,26,38,38],[32,32,38,38],[34,19,39,38],[34,31,41,41],[33,32,41,40],[24,28,41,41],[32,33,41,38],[33,32,39,39],[19,19,17,18],[19,19,17,17],[19,22,27,25],[18,19,25,28],[22,27,19,19],[32,28,20,27]]

plan_2024=[[14,0,0,0],[12, 0, 0, 0],[ 2, 0, 0, 0],[ 8, 0, 0, 0],[ 5, 0, 0, 0],[10, 0, 0, 0],
           [15, 0, 0, 0],[ 9, 0, 0, 0],[13, 0, 0, 0],[ 9, 0, 0, 0],[15, 0, 0, 0],[ 3, 0, 0, 0],
           [12, 0, 0, 0],[ 4, 0, 0, 0],[ 8, 0, 0, 0],[ 9, 0, 0, 0],[14, 0, 0, 0],[12, 0, 0, 0],
           [ 8, 0, 0, 0],[ 2, 0, 0, 0],[ 2, 0, 0, 0],[11, 0, 0, 0],[ 7, 0, 0, 0],[ 4, 0, 0, 0],
           [10, 0, 0, 0],[ 8, 0, 0, 0],[27,34,37,37],[16,24,36,37],[32,27,36,36],[21,23,36,36],
           [24,32,37,37],[24,33,35,36],[19,21,37,35],[26,25,36,37],[25,28,41,39],[32,34,39,41],
           [34,32,39,41],[34,17,40,38],[23,30,38,40],[20,17,38,41],[24,31,39,41],[26,22,41,39],
           [25,21,41,41],[19,23,38,38],[34,21,39,38],[26,17,38,39],[23,19,39,40],[18,22,38,40],
           [26,29,38,40],[22,27,40,40],[17,23,18,28],[20,30,21,27],[31,24,34,33],[22,21,18,31]]


plan_2023=[[6,0,0,0],[7,0,0,0],[7,0,0,0],[1,0,0,0],[4,0,0,0],[8,0,0,0],
           [6,0,0,0],[2,0,0,0],[3,0,0,0],[4,0,0,0],[5,0,0,0],[8,0,0,0],
           [6,0,0,0],[8,0,0,0],[9,0,0,0],[10,0,0,0],[1,0,0,0],[7,0,0,0],
           [14,0,0,0],[15,0,0,0],[11,0,0,0],[12,0,0,0],[1,0,0,0],[13,0,0,0],
           [6,0,0,0],[3,0,0,0],[20,20,36,36],[28,28,35,35],[21,21,35,35],[22,22,35,35],
           [17,17,36,36],[18,18,37,37],[16,0,0,0],[16,0,0,0],[18,18,38,38],[24,24,38,38],
           [25,25,38,38],[26,26,39,39],[28,28,39,39],[27,27,39,39],[19,19,40,40],[19,19,40,40],
           [18,18,40,40],[17,17,41,41],[17,17,41,41],[22,22,41,41],[21,21,41,41],[29,29,41,41],
           [30,27,41,41],[31,31,41,41],[32,33,24,21],[25,26,22,29],[17,17,28,30],[19,19,34,23]]

#2023亩产量，产品编号*土地类型
yield_2023=[[400,380,360,0,0,0],[500,475,450,0,0,0],[400,380,360,0,0,0],[350,330,315,0,0,0],[415,395,375,0,0,0],[800,760,720,0,0,0]
    ,[100,950,900,0,0,0],[400,380,360,0,0,0],[630,600,570,0,0,0],[525,500,475,0,0,0],[110,105,100,0,0,0],[3000,2850,27000,0,0]
    ,[2200,2100,2000,0,0,0],[420,400,380,0,0,0],[525,500,475,0,0,0],[0,0,0,500,0,0],[0,0,0,3000,3600,3200],[0,0,0,2000,2400,2200]
    ,[0,0,0,3000,3600,3200],[0,0,0,2000,2400,2200],[0,0,0,2400,3000,2700],[0,0,0,6400,8000,7200],[0,0,0,2700,3300,3000],[0,0,0,2400,3000,2700]
    ,[0,0,0,3300,4000,3600],[0,0,0,3700,4500,4100],[0,0,0,4100,5000,4500],[0,0,0,3200,4000,3600],[0,0,0,12000,15000,13500],[0,0,0,4100,5000,4500]
    ,[0,0,0,1600,2000,1800],[0,0,0,10000,12000,11000],[0,0,0,5000,6000,5400],[0,0,0,5500,6600,6000],[0,0,0,5000,0,0],[0,0,0,4000,0,0]
    ,[0,0,0,3000,0,0],[0,0,0,0,5000,0],[0,0,0,0,4000,0],[0,0,0,0,10000,0],[0,0,0,0,1000,0]]

#2023种植成本，产品编号*土地类型
cost_2023=[[400,400,400,0,0,0],[400,400,400,0,0,0],[350,350,350,0,0,0],[350,350,350,0,0,0],[350,350,350,0,0,0],[450,450,450,0,0,0],
    [500,500,500,0,0,0],[360,360,360,0,0,0],[400,400,400,0,0,0],[360,360,360,0,0,0],[350,350,350,0,0,0],[100,1000,1000,0,0,0],
    [200,2000,2000,0,0,0],[400,400,400,0,0,0],[350,350,350,0,0,0],[0,0,0,680,0,0],[0,0,0,2000,2400,2640],[0,0,0,1000,1200,1320],
    [0,0,0,2000,2400,2640],[0,0,0,2000,2400,2640],[0,0,0,2000,2400,2640],[0,0,0,2000,2400,2640],[0,0,0,2300,2700,3000],[0,0,0,1600,2000,2200],
    [0,0,0,2400,3000,3300],[0,0,0,2900,3500,3850],[0,0,0,1600,2000,2200],[0,0,0,1600,2000,2200],[0,0,0,2900,3500,3850],[0,0,0,1600,2000,2200],
    [0,0,0,1000,1200,1300],[0,0,0,4100,5000,5500],[0,0,0,2000,2500,2750],[0,0,0,900,1100,1200],[0,0,0,2000,0,0],[0,0,0,500,0,0],[0,0,0,500,0,0],
    [0,0,0,0,3000,0],[0,0,0,0,2000,0],[0,0,0,0,10000,0],[0,0,0,0,10000,0]]

#2023种植销售单价，产品编号*土地类型
price_2023=[[[2.50,4.00],[2.50,4.00],[2.50,4.00],[0,0],[0,0],[0,0]],
            [[6.50,8.50],[6.50,8.50],[6.50,8.50],[0,0],[0,0],[0,0]],
            [[7.50,9.00],[7.50,9.00],[7.50,9.00],[0,0],[0,0],[0,0]],
            [[6.00,8.00],[6.00,8.00],[6.00,8.00],[0,0],[0,0],[0,0]],
            [[6.00,7.50],[6.00,7.50],[6.00,7.50],[0,0],[0,0],[0,0]],
            [[3.00,4.00],[3.00,4.00],[3.00,4.00],[0,0],[0,0],[0,0]],
            [[2.50,3.50],[2.50,3.50],[2.50,3.50],[0,0],[0,0],[0,0]],
            [[6.00,7.50],[6.00,7.50],[6.00,7.50],[0,0],[0,0],[0,0]],
            [[5.50,6.50],[5.50,6.50],[5.50,6.50],[0,0],[0,0],[0,0]],
            [[6.50,8.50],[6.50,8.50],[6.50,8.50],[0,0],[0,0],[0,0]],
            [[30.00,50.00],[30.00,50.00],[30.00,50.00],[0,0],[0,0],[0,0]],
            [[1.00,2.00],[1.00,2.00],[1.00,2.00],[0,0],[0,0],[0,0]],
            [[2.50,4.00],[2.50,4.00],[2.50,4.00],[0,0],[0,0],[0,0]],
            [[5.00,6.00],[5.00,6.00],[5.00,6.00],[0,0],[0,0],[0,0]],
            [[3.00,4.00],[3.00,4.00],[3.00,4.00],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[6.00,8.00],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[7.00,9.00],[7.00,9.00],[8.40,10.80]],
            [[0,0],[0,0],[0,0],[5.50,8.00],[5.50,8.00],[6.60,9.60]],
            [[0,0],[0,0],[0,0],[5.00,8.00],[5.00,8.00],[6.00,9.60]],
            [[0,0],[0,0],[0,0],[3.00,4.50],[3.00,4.50],[3.60,5.40]],
            [[0,0],[0,0],[0,0],[5.00,7.50],[5.00,7.50],[6.00,9.00]],
            [[0,0],[0,0],[0,0],[5.00,6.00],[5.00,6.00],[6.00,7.20]],
            [[0,0],[0,0],[0,0],[4.80,6.70],[4.80,6.70],[5.80,8.00]],
            [[0,0],[0,0],[0,0],[4.00,6.50],[4.00,6.50],[5.80,7.80]],
            [[0,0],[0,0],[0,0],[5.00,6.00],[5.00,6.00],[6.00,7.20]],
            [[0,0],[0,0],[0,0],[5.50,7.50],[5.50,7.50],[6.60,9.00]],
            [[0,0],[0,0],[0,0],[4.00,6.00],[4.00,6.00],[4.80,7.20]],
            [[0,0],[0,0],[0,0],[5.00,6.50],[5.00,6.50],[6.00,7.80]],
            [[0,0],[0,0],[0,0],[6.00,8.00],[6.00,8.00],[7.20,9.60]],
            [[0,0],[0,0],[0,0],[4.50,6.00],[4.50,6.00],[5.40,7.20]],
            [[0,0],[0,0],[0,0],[6.00,8.50],[6.00,8.50],[7.20,10.20]],
            [[0,0],[0,0],[0,0],[3.00,6.00],[3.00,6.00],[3.60,7.20]],
            [[0,0],[0,0],[0,0],[4.00,5.00],[4.00,5.00],[4.80,6.00]],
            [[0,0],[0,0],[0,0],[3.20,4.80],[3.20,4.80],[3.80,5.80]],
            [[0,0],[0,0],[0,0],[2.00,3.00],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[2.00,3.00],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[2.50,4.00],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[50.00,65.00],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[18.00,20.00],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[14.00,18.00],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[80.00,120.00],[0,0]]]

#遗传算法寻找最优解
#创建种群个体
creat(*field)
creat(*field1)
creat(*field2)
creat(*field3)
creat(*field4)
creat(*field5)
fitness=[0,0,0,0,0,0] #存储各个个体的适应度
Gtime=200#迭代次数
q_possibility=0.7 #发生交叉的概率
p_possibility=0.1 #发生变异的概率
fitBest=0 #记录最优适应度
for i in range(Gtime):
    fitness[0] = target_1(7,*field)
    fitness[1] = target_1(7,*field1)
    fitness[2] = target_1(7,*field2)
    fitness[3] = target_1(7,*field3)
    fitness[4] = target_1(7,*field4)
    fitness[5] = target_1(7,*field5)
    maxfit=max(fitness)
    minfit=min(fitness)
    for j in range(len(fitness)):
        if fitness[j]==maxfit and maxfit>fitBest:
            fitBest=maxfit
            findex=j #存储当前最优解所在位置
    for j in range(len(fitness)):
        if fitness[j]==minfit:
            fDiscard=j #存储当前最劣解所在位置

    if findex==0 and maxfit==fitBest:
        fBest=field
    elif findex==1 and maxfit==fitBest:
        fBest=field1
    elif findex==2 and maxfit==fitBest:
        fBest=field2
    elif findex==3 and maxfit==fitBest:
        fBest=field3
    elif findex==4 and maxfit==fitBest:
        fBest=field4
    elif findex == 5 and maxfit == fitBest:
        fBest=field5

    if fDiscard==0:#舍弃最劣的解，用新的种群个体替换
        init_field(*field)
        creat(*field)
    elif fDiscard==1:
        init_field(*field1)
        creat(*field1)
    elif fDiscard==2:
        init_field(*field2)
        creat(*field2)
    elif fDiscard==3:
        init_field(*field3)
        creat(*field3)
    elif fDiscard==4:
        init_field(*field4)
        creat(*field4)
    else:
        init_field(*field5)
        creat(*field5)

    #交叉操作
    q=random.random()
    if q<q_possibility:
        x=random.randint(0,53)
        for m in range(4):
            temp=field[x][m]
            field[x][m]=field1[x][m]
            field1[x][m]=temp
    q = random.random()
    if q < q_possibility:
        x = random.randint(0, 53)
        for m in range(4):
            temp = field1[x][m]
            field1[x][m] = field2[x][m]
            field2[x][m] = temp
    q = random.random()
    if q < q_possibility:
        x = random.randint(0, 53)
        for m in range(4):
            temp = field2[x][m]
            field2[x][m] = field3[x][m]
            field3[x][m] = temp
    q = random.random()
    if q < q_possibility:
        x = random.randint(0, 53)
        for m in range(4):
            temp = field3[x][m]
            field3[x][m] = field4[x][m]
            field4[x][m] = temp
    q = random.random()
    if q < q_possibility:
        x = random.randint(0, 53)
        for m in range(4):
            temp = field4[x][m]
            field4[x][m] = field5[x][m]
            field5[x][m] = temp
    #print(fitBest)

    #变异操作
    p = random.random()
    if p < p_possibility:
        x = random.randint(0, 53)
        if x>26:
            for m in range(4):
                field[x][m]=random.randint(1,41)
                while judge(field[x][5],field[x][m],plan_2023[x][0],plan_2023[x][1],plan_2023[x][2],plan_2023[x][3],1)==False:
                    field[x][m] = random.randint(1, 41)
        else:
            field[x][0] = random.randint(1, 41)
            while judge(field[x][5], field[x][0], plan_2023[x][0], plan_2023[x][1], plan_2023[x][2], plan_2023[x][3],1) == False:
                field[x][0] = random.randint(1, 41)
    if p < p_possibility:
        x = random.randint(0, 53)
        if x>26:
            for m in range(4):
                field1[x][m]=random.randint(1,41)
                while judge(field1[x][5],field1[x][m],plan_2023[x][0],plan_2023[x][1],plan_2023[x][2],plan_2023[x][3],1)==False:
                    field1[x][m] = random.randint(1, 41)
        else:
            field1[x][0] = random.randint(1, 41)
            while judge(field1[x][5], field1[x][0], plan_2023[x][0], plan_2023[x][1], plan_2023[x][2], plan_2023[x][3],1) == False:
                field1[x][0] = random.randint(1, 41)
    if p < p_possibility:
        x = random.randint(0, 53)
        if x>26:
            for m in range(4):
                field2[x][m]=random.randint(1,41)
                while judge(field2[x][5],field2[x][m],plan_2023[x][0],plan_2023[x][1],plan_2023[x][2],plan_2023[x][3],1)==False:
                    field2[x][m] = random.randint(1, 41)
        else:
            field2[x][0] = random.randint(1, 41)
            while judge(field2[x][5], field2[x][0], plan_2023[x][0], plan_2023[x][1], plan_2023[x][2], plan_2023[x][3],1) == False:
                field2[x][0] = random.randint(1, 41)
    if p < p_possibility:
        x = random.randint(0, 53)
        if x>26:
            for m in range(4):
                field3[x][m]=random.randint(1,41)
                while judge(field3[x][5],field3[x][m],plan_2023[x][0],plan_2023[x][1],plan_2023[x][2],plan_2023[x][3],1)==False:
                    field3[x][m] = random.randint(1, 41)
        else:
            field3[x][0] = random.randint(1, 41)
            while judge(field3[x][5], field3[x][0], plan_2023[x][0], plan_2023[x][1], plan_2023[x][2], plan_2023[x][3],1) == False:
                field3[x][0] = random.randint(1, 41)
    if p < p_possibility:
        x = random.randint(0, 53)
        if x>26:
            for m in range(4):
                field4[x][m]=random.randint(1,41)
                while judge(field4[x][5],field4[x][m],plan_2023[x][0],plan_2023[x][1],plan_2023[x][2],plan_2023[x][3],1)==False:
                    field4[x][m] = random.randint(1, 41)
        else:
            field4[x][0] = random.randint(1, 41)
            while judge(field4[x][5], field4[x][0], plan_2023[x][0], plan_2023[x][1], plan_2023[x][2], plan_2023[x][3],1) == False:
                field4[x][0] = random.randint(1, 41)
    if p < p_possibility:
        x = random.randint(0, 53)
        if x>26:
            for m in range(4):
                field5[x][m]=random.randint(1,41)
                while judge(field5[x][5],field5[x][m],plan_2023[x][0],plan_2023[x][1],plan_2023[x][2],plan_2023[x][3],1)==False:
                    field5[x][m] = random.randint(1, 41)
        else:
            field5[x][0] = random.randint(1, 41)
            while judge(field5[x][5], field5[x][0], plan_2023[x][0], plan_2023[x][1], plan_2023[x][2], plan_2023[x][3],1) == False:
                field5[x][0] = random.randint(1, 41)
    print(fitBest)
print(fBest)
# a=cost(* field)
# print(a)
# a=profit_cal(* field)
# print(a)
# a=target_1(*field)
# print(a)
