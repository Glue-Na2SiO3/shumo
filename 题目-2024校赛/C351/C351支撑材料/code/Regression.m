[b,bint,r,rint,stats]=regress(Y,X,0.05)
disp(b)
rcoplot(r,rint)
disp(stats)