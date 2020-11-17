function [gamser,gln] = gser(a,x)
% aus NR
% 
% returens the incomplete gammma function P(a,x) evaluated by its series
% representation as gamser. Also returns ln Gamma(a) as gln
%
ITMAX = 100;
EPS = 3e-7; 
gln = gammln(a); 
if x <= 0
    if (x < 0) 
         gamser = 0; 
    end 
end 
% 
ap = a; 
sum = 1/a;
del = sum; 
 for n = 1 : ITMAX
     ap = ap +1; 
     del = del*x/ap; 
     sum = sum + del; 
     if abs(del) < abs(sum)*EPS 
         continue 
     end 
  end    
  gamser = sum*exp(-x+a*log(x)-gln); 


    