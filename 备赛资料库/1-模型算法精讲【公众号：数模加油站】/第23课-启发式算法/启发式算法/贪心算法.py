from util import *

with open("points.json","r") as f:
    txt=f.readline()
points=json.loads(txt)

N=len(points)

ans=greedy(2,points)
print(ans)
print(calc(ans,points))

plot_ans(ans,points)
