% y = a + bx
zet_0 = 20; 
k = 0; 
%
  fit_breite = 10  ; % 10 rbins = 75 m 
  sig = 0; 
  ndata = 2*fit_breite + 1; 
  for i = zet_0 + fit_breite : rbins - fit_breite
      x = range(i-fit_breite : i+fit_breite); 
      y = log_Raman(i-fit_breite : i+fit_breite); 
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x; 
    box on 
    hold on
    plot(x,y,'b')
    plot(x,yy,'c')
    
 ableitung_chi2(i) = (yy(2*fit_breite) - yy(1))/...
                        (x(2*fit_breite)- x(1));
  end