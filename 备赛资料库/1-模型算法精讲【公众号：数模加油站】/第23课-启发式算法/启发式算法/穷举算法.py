import itertools,time
from util import *

with open("points.json","r") as f:
    txt=f.readline()
points=json.loads(txt)

N=len(points)

s=time.time()

nums = list(range(1,N))
permutations = list(itertools.permutations(nums))
min_len=1e10
for route in permutations:
    route=list(route)
    route.insert(0,0)
    l = calc(route,points)
    if l<min_len:
        ans=route
        min_len=l

t=time.time()

print(min_len)
print("time:"+str(t-s))
print(ans)
plot_ans(ans,points)