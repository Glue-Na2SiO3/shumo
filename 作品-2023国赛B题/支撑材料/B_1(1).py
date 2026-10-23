import math
D=90.948
degree=120
alpha=1.5
d=200 #船距海域中心点处的距离


#B题第一问
def Cal_W(alpha,D,degree): #计算覆盖宽度
    W=math.cos(math.radians(alpha))*D*math.sin(math.radians(degree/2))*(1/math.sin(math.radians(90-alpha-degree/2))+1/math.sin(math.radians(90+alpha-degree/2)))
    return W

def Upper_W(alpha,D,degree):
    W=math.cos(math.radians(alpha))*D*math.sin(math.radians(degree/2))*1/math.sin(math.radians(90+alpha-degree/2))
    return W

def Lowwer_W(alpha,D,degree):
    W=math.cos(math.radians(alpha))*D*math.sin(math.radians(degree/2))*1/math.sin(math.radians(90-alpha-degree/2))
    return W

def Cal_n(alpha,D1,D2,degree,d): #计算重叠率(由下向上) D1>D2
    W1=math.cos(math.radians(alpha))*D1*math.sin(math.radians(degree/2))*1/math.sin(math.radians(90+alpha-degree/2))
    W2=math.cos(math.radians(alpha))*D2*math.sin(math.radians(degree/2))*1/math.sin(math.radians(90-alpha-degree/2))
    n=(W1+W2-d)/(W1+W2)
    return n
#print(Cal_n(1.5,90.948,85.712,120,200))

#B题第二问

#print(math.atan(math.sqrt(1))/math.pi*180)

# beta=[0,45,90,135,180,225,270,315]
# distance=[0,0.3,0.6,0.9,1.2,1.5,1.8,2.1]#船距海域中心点处的距离
# for i in range (len(beta)):
#     for j in range(len(distance)):
#         tan_gamma = math.tan(math.radians(alpha)) * math.sin(math.radians(beta[i]))
#         gamma = math.atan(tan_gamma) / math.pi * 180
#         D=120-tan_gamma*distance[j]*1852 #海里和米的单位转化
#         print(Cal_W(gamma,D,degree))


# print(tan_gamma)
# print(gamma)
#根据上一问，取随船距海域中心点处的距离（+）变化，d递减


#B题第三问 (南北方向行驶)

width=2*1852
length=4*1852
D=110
cel_edge0=length/2+math.cos(math.radians(alpha))*D*math.sin(math.radians(degree/2))*1/math.sin(math.radians(90+alpha-degree/2)) #覆盖范围的上边沿，由海域中心开始
flo_edge0=length/2-math.cos(math.radians(alpha))*D*math.sin(math.radians(degree/2))*1/math.sin(math.radians(90-alpha-degree/2))  #覆盖范围的下边沿，由海域中心开始
# print(cel_edge0)
# print(flo_edge0)
cel_edge=cel_edge0
#print(cel_edge)
flo_edge=flo_edge0

# d0=1
# while cel_edge<length: #先向上寻找
#     D_temp=D-math.tan(math.radians(alpha))*1*d0  #以1m步长寻找10%重叠率的间隔位置
#     d0+=1
#     if (abs(Cal_n(alpha,D,D_temp,degree,d0))-0.1<0.012 and Cal_n(alpha,D,D_temp,degree,d0) >0.1) :
#         #print(Cal_n(alpha,D,D_temp,degree,d0))
#         # print("this is n")
#         # print(Cal_n(alpha,D,D_temp,degree,d0))
#         D=D_temp
#         #print("this is D")
#         #print(D)
#         #print("this is d0")
#         #print(d0)
#         cel_edge+=d0
#         print(cel_edge-cel_edge0+length/2)
#         d0=1
#         #print("------------")
# #print("end!")

# d0=1
# while flo_edge>0: #再向下寻找
#     D_temp=D+math.tan(math.radians(alpha))*1*d0  #以1m步长寻找10%重叠率的间隔位置
#     d0+=1
#     if (abs(Cal_n(alpha,D_temp,D,degree,d0))-0.1<0.012 and Cal_n(alpha,D_temp,D,degree,d0) >0.1) :
#         #print(Cal_n(alpha,D,D_temp,degree,d0))
#         # print("this is n")
#         # print(Cal_n(alpha,D,D_temp,degree,d0))
#         D=D_temp
#         #print("this is D")
#         #print(D)
#         #print("this is d0")
#         #print(d0)
#         flo_edge-=d0
#         print(flo_edge-flo_edge0+length/2)
#         d0=1
#         #print("------------")
#print("end!")

#第四问漏测面积计算
# for i in range(length):
#     for j in range(width):
#         a_list = [1.3, 1.4, 1.5]
#     for i in range(len(a_list)):
#         a = a_list[i]
#         b = 120
#         radian1 = a * math.pi / 180
#         radian2 = b / 2 * math.pi / 180
#         D0 = 110
#         Les = 4 * 1852  # 待测海域东西宽度
#         D = D0 + Les / 2 * math.tan(radian1)  # 此处Dmax
#         x = D * math.tan(radian2)
#         D = D - x * math.tan(radian1)
#         W1 = 0
#         W2 = 0
#         res_list = [x]
#         left_list = []
#         right_list = []
#
#         while x + W2 * math.cos(radian1) <= Les:
#             W2 = D * math.sin(radian2) / math.cos(radian2 - radian1)
#             W1 = D * math.sin(radian2) / math.cos(radian2 + radian1)
#             d = (W1 + W2) * 0.9 * math.cos(radian1) / (
#                         1 + math.sin(radian1) * math.sin(radian2) / math.cos(radian2 + radian1) * 0.9)
#             left_list.append(x - W1 * math.cos(radian1))
#             right_list.append(x + W2 * math.cos(radian1))
#             D = D - d * math.tan(radian1)
#             x = x + d
#             res_list.append(x)
#
#         W2 = D * math.sin(radian2) / math.cos(radian2 - radian1)
#         W1 = D * math.sin(radian2) / math.cos(radian2 + radian1)
#         d = (W1 + W2) * 0.9 * math.cos(radian1) / (
#                     1 + math.sin(radian1) * math.sin(radian2) / math.cos(radian2 + radian1) * 0.9)
#         plt.plot(range(len(res_list)), res_list, linestyle='-', label=f'坡角为{a}°')
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
#     plt.rcParams['axes.unicode_minus']

