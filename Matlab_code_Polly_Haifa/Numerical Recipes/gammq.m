function [erg,gammq] = gammq(a,x)
% 
% returen the incomlete gann function Q(q,x) = 1-P(a,x) 
% aus NR                    to Matlab  09/06  BHeese
% 
if(x < 0 || a < 0)          % bad arguments in 'gammaq'
    return 
end
    if (x < (a+1))          % use the series representation 
      erg = gser(a,x);  
      gammq = 1-gamser;     % and takes its complement
    else                    % use the continued fraction represantation 
       erg =  ggcf(a,x); 
       gammq = gammcf; 
    end 
    