import math,json
import matplotlib.pyplot as plt

def calc(route,points):
    N=len(route)
    l = 0
    for i in range(N-1):
        l += math.sqrt(
            (points[route[i + 1]][0] - points[route[i]][0]) ** 2 + (points[route[i + 1]][1] - points[route[i]][1]) ** 2)
    l+=math.sqrt((points[route[0]][0] - points[route[-1]][0]) ** 2 + (points[route[0]][1] - points[route[-1]][1]) ** 2)
    return l

def plot_ans(route,points):
    x = []
    y = []
    for i in route:
        x.append(points[i][0])
        y.append(points[i][1])
    x.append(points[route[0]][0])
    y.append(points[route[0]][1])
    plt.plot(x, y)
    plt.show()

def greedy(start,points):
    N=len(points)
    visited = [False] * N
    visited[start] = True
    ans = [start]

    for i in range(N - 1):
        nxt = 0
        min_len = 1e9
        for j in range(N):
            if not visited[j]:
                l = math.sqrt((points[ans[-1]][0] - points[j][0]) ** 2 + (points[ans[-1]][1] - points[j][1]) ** 2)
                if l < min_len:
                    nxt = j
                    min_len = l
        ans.append(nxt)
        visited[nxt] = True
    #ans.append(start)
    return ans