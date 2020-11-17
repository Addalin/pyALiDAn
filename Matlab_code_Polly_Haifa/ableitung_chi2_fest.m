function [abl_chi2,chi2,q,u] = ableitung_chi2_fest(input,range,alt,rbbb,fit_breite,datum2);
%------------------------------------------------------------------
% Ableitung berechnen                           09/06   BHeese
% für Polly und PollyXT angeglichen             04/07   BHeese
% Am Ende rbbb-1, da sonst ndata zu klein ist   08/07 BHeese 
% kosmetische Behandlung			10/07 	BHeese	
%
% mit zunehmender Höhe gleiche Anzahl Bins zum mitteln verwenden
%
 warning off all
% unten = 23; % tags
% unten =  6; % nachts 080707 00-02
% unten =  10; % abends ab dann ist log_pr2_sm ansteigend  080707
%  unten = 23; % 070923 nachts 0205-0405
% unten = 1; % 070923 nachts 0205-0405 neu am 16.10.07
 unten = 12; % Polly Korea
  oben = 500; 
%oben = 98; % RefBin(2) 070807 morgens
u=unten; 
clear x y
%
deltar = range(2)-range(1); 

% --------------------------------------
%    fit to a staight line y = a + bx
% --------------------------------------
mwt = 0;
sig = 0;  
%  
%   Protokol schreiben
% ---------------------
  O = [ unten , fit_breite ];
  dlmwrite(['protokol_linfit_' datum2 '.dat'], O,'delimiter', '\t');  
  N = ['i' 'ndata'];
  dlmwrite(['protokol_linfit_' datum2 '.dat'], N,'delimiter', '\t', '-append');
%  
% ---------------------------------------- 
%  am unteren Rand variable Fitbreite 
% ----------------------------------------
  for i = unten : unten + fit_breite
      ndata = (2*(i-unten)+1); 
      % soll bei unten = 10 anfangen und nur nach oben hin fitten 
      % 
      % schreiben
          M = [alt(i),ndata, ndata*deltar];
          dlmwrite(['protokol_linfit_' num2str(datum2) '.dat'], M, 'delimiter', '\t', '-append'); 
      %    
      x = range(i-(i-unten):i+(i-unten)+1); 
      y = input(i-(i-unten):i+(i-unten)+1); 
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x;   
    figure(1)
    box on 
    hold on
    plot(x,y,'r')
    plot(x,yy,'g')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata) - x(1));
  end
% ------------------------------
%      unten
% ------------------------------
  for i = unten + fit_breite+1 : oben - fit_breite  
      ndata = 2*fit_breite+1;
      % schreiben
          M = [alt(i),ndata , ndata*deltar];
          dlmwrite(['protokol_linfit_' num2str(datum2) '.dat'], M, 'delimiter', '\t', '-append'); 
      %    
      x = range(i-fit_breite+1 : min(i+fit_breite+1,rbbb));
      y = input(i-fit_breite+1 : min(i+fit_breite+1,rbbb));
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x; 
    plot(x,y,'b')
    plot(x,yy,'c')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata)- x(1));
        if i >= 0.8*rbbb
            return
        end
  end
