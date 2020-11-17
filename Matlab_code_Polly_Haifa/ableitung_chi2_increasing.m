function [abl_chi2,chi2,q,u] = ableitung_chi2_increasing(input,range,rbbb,fit_width,datum2)
%------------------------------------------------------------------
% Ableitung berechnen           		09/06   BHeese
% für Polly und PollyXT angeglichen     04/07   BHeese
% kosmetische Behandlung			    10/07 	BHeese	
% 
% (mit zunehmender Höhe über mehr Bins mitteln)
%  Änderungen über k und ndata! 
%
warning off all
clear x y
%unten = 24; war vor 18.1.2008 eingestellt. 
%unten = 88; % Hefei Lidar 03.12.2009
%unten = 2; % Hefei Lidar 04.12.2009
unten =12; % Korea 2.2.2011
%unten =12; % Korea 27.10.2010
u = unten; 
%
% --------------------------------------
%    fit to a staight line y = a + bx
% --------------------------------------
mwt = 0;
sig = 0;  
%
deltar = range(2)-range(1); 
%
% ----------------------
%   Protokol schreiben
% ----------------------
  O = [ unten , fit_width ];
  dlmwrite(['protokol_linfit_' datum2 '.dat'], O,'delimiter', '\t');  
  N = ['i' 'ndata'];
  dlmwrite(['protokol_linfit_' datum2 '.dat'], N,'delimiter', '\t', '-append');
%
% --------------------------------------
%  am unteren Rand variable Fitbreite
% --------------------------------------
  for i = unten : unten + fit_width
      ndata = (2*(i-unten)+1); 
      % soll bei unten = 10 anfangen und nur nach oben hin fitten 
      % schreiben
          M = [range(i),ndata, ndata*deltar];
          dlmwrite(['protokol_linfit_' datum2 '.dat'], M, 'delimiter', '\t', '-append'); 
      %    
      x = range(i-(i-unten):i+(i-unten)+1); 
      y = input(i-(i-unten):i+(i-unten)+1); 
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x;   
   figure(1)   
    box on 
    hold on
    plot(y,x,'r')
    plot(yy,x,'g')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata) - x(1));
  end
% -----------------------
%      Mittelteil
% -----------------------
  for i = unten + fit_width+1 : rbbb - fit_width  
      k = fix(0.1*i);
      ndata = 2*fit_width+1+2*k;
      % schreiben
          M = [range(i),ndata , ndata*deltar];
          dlmwrite(['protokol_linfit_' datum2 '.dat'], M, 'delimiter', '\t', '-append'); 
      %    
      x = range(i-fit_width+1-k : min(i+fit_width+1+k,rbbb));
      y = input(i-fit_width+1-k : min(i+fit_width+1+k,rbbb));
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x; 
    plot(y,x,'b')
    plot(yy,x,'c')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata)- x(1));
        if i >= 0.8*rbbb
            return
        end
  end
%
%
%  am oberen Rand    % braucht nicht - ist eh verrauscht
% -----------------
%  for i=rbbb-fit_breite : rbbb
%    x = range(i:i+fit_breite+1); 
%      y = input(i:i+fit_breite+1); 
% [a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
%    yy = a + b.*x; 
%    box on 
%    hold on
%    plot(x,y,'b')
%    plot(x,yy,'c')
%    
% abl_chi2(i) = (yy(fit_breite+1) - yy(i))/...
%                        (x(fit_breite+1)- x(i));
%  end
% 
