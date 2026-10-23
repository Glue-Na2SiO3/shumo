function yhat=logisfun(beta,x)
yhat=beta(3)./(1+(beta(3)./beta(1)-1).*exp(-beta(2)*x));
