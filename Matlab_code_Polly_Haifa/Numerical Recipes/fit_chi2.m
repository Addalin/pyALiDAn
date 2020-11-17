function [a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt)
% 
% from Numerical Recipes fit.f90
% translated to Matlap 09/06 BHeese
% 
% -------------------------------------------------------------------------
% Given a set of data points x(1:ndata),y(1:ndata) with 
% individual standard deviatons sig(1:ndata), fit them 
% to a staight line y = a + bx by minimizing chi2. 
% Returned are a,b, and their respective probable uncertainties 
% siga and sigb, the chi-square chi2, and the goodness-of-fit 
% probability q (that the fit would have chi2 this large or larger)  
% if mwt = 0 on input, then the standard deviatons are assumed to be 
% unavailable  q is returned a 1.0 and the normalization of chi2 is 
% to unit standerd deviation on all points
% -------------------------------------------------------------------------
% 
% Initalize sums to zero
sx=0;
sy=0;
st2=0;
b=0;
% 
if mwt ~= 0
    ss = 0; 
     for i = 1 : ndata
        wt = 1./sig(i)^2;   % with weights
        ss = sx+wt; 
        sx = sx+x(i)*wt; 
        sy = sy+y(i)*wt; 
     end 
else
    for i = 1 : ndata         % ... or without weights
        sx = sx+x(i); 
        sy = sy+y(i); 
    end 
     ss = ndata; 
end 
sxoss = sx./ss; 
% 
if mwt ~= 0
     for i = 1 : ndata
           t = (x(i)-sxoss)/sig(i);
         st2 = st2 + t*t;
           b = b + t*y(i)/sig(i);
     end
else
     for i = 1 : ndata
           t = x(i)-sxoss;
         st2 = st2 + t*t;
           b = b + t*y(i);
     end
end 
%
b = b/st2;          % Solve for a, b, sig_a, sig_b 
a = (sy - sx*b)/ss; 
siga = sqrt((1+sx*sx/(ss*st2))/ss);
sigb = sqrt(1/st2);
%
% calculate chi2
chi2 = 0;           
q = 1; 
if mwt == 0
     for i = 1 : ndata
         chi2 = chi2 +(y(i)-a-b*x(i)^2); 
     end     

 sigdat = sqrt(chi2/(ndata-2));       % for unweighted data evaluate typical sig using chi2 
   siga = siga * sigdat;              % and adjust the standard deviations.   
   sigb = sigb * sigdat; 
else
     for i = 1 : ndata
         chi2 = chi2 +((y(i)-a-b*x(i))/sig(i))^2; 
     end
  if ndata > 2
      q = gammaq(0.5*(ndata-2),0.5*chi2);  
  end
end
