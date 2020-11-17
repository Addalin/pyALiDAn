function [gammacf,gln] = gcf(a,x)
% aus NR
% returens the incomplete gammma function Q(a,x) evaluated by its
% continued raction representations as gammacf.
% Also returns ln Gamma(a) as gln. 
% 
% Parameter
ITMAX = 100;            % maximum allowed number of iterations 
EPS = 3e-7;             % relative accuracy
FPMIN = 1e-30;          % is a number near the smallest floating-point number
%  
gln = gammln(a); 
b= x+1-a;      % Set up for evaluating continued fra<ctions by modified Lentz's method with b0=0. 
c = 1/FPMIN; 
d=1/b; 
h=d;
%
for i = 1 : ITMAX
     an = -i*(i-a);
     b=b+2;
     d=an*d+b;
     if (abs(d)) < PMIN 
         d = FPMIN;
     end 
     c=b+an/c;
      if (abs(c)) < PMIN 
         c = FPMIN;
      end 
     d = 1./d;
     del = d*c;
     h = h*del;
     if (abs(del-1)) < EPS 
       continue
     end 
end  
 gammacf = exp(-x+a*log(x)-gln)*h;       % put factors in front