function [abl_chi2,chi2,q,u] = ableitung_chi2_fest(input,range,alt,rbbb,fit_breite,datum1);
%------------------------------------------------------------------
% Ableitung berechnen                           09/06   BHeese
% für Polly angeglichen                         04/07   BHeese
% Am Ende rbbb-1, da sonst ndata zu klein ist   08/07 	BHeese 
% kosmetische Behandlung			10/07 	BHeese	
%
% mit zunehmender Höhe die Anzahl Bins zum mitteln verdoppeln,
% immer dann, wenn signal anfängt zu mäandern (Poisson-Statistik) 
%
 warning off all
%**********
%  070807 
%**********
%070807 abends 
%  unten = 4; ab dann ist log_pr2_sm ansteigend  080707 0000-0200h 
% mitte1 = 25; 
% mitte2 = 49; 
%   oben = 100; 
%cirren_unten = 257;
%cirren_oben =  326; 
%
%***********
%  070923
%***********
% unten = 20; % ab dann ist log_pr2_sm ansteigend 0000-0200h 
% u = unten; 
%% unten = 10;
%mitte1 = 60; 
%mitte2 = 100; 
%  oben = 200; 
%cirren_unten = 257;
%cirren_oben =  326; 
%**************
%  071204 FMI
%**************
% unten = 24; % ab dann ist log_pr2_sm ansteigend 0000-0200h 
% u = unten; 
%mitte1 = 60; 
%mitte2 = 100; 
%  oben = 200; 
%
%******************  
% 080321 FMI abends
%******************
%unten = 33; %ab dann ist log_pr2_sm ansteigend  080321 ab 1800 h 
%mitte1 = 75; 
%mitte2 = 100; 
%  oben = 200; 
%
%******************
% 080609 Melpitz nachts
%******************
%unten = 20;
%mitte1 = 50; 
%mitte2 = 100; 
%  oben = 200; 
%
%*****************
%  080715 Melpitz nachts
%*****************
%unten = 11;
%mitte1 = 50; 
%mitte2 = 70; 
%  oben = 90; 
%PollyXT  
 unten = 32;
mitte1 = 1000; 
mitte2 = 1000; 
  oben = 1000; 
%  
% Hefei Lidar
% unten = 62;
%mitte1 = 200; 
%mitte2 = 400; 
%  oben = 800; 
%  
u = unten;
% -----------start----------------------
clear xx x y
deltar = range(2)-range(1);
% --------------------------------------
%    fit to a staight line y = a + bx
% --------------------------------------
mwt = 0;
sig = 0;  
%  
%   Protokol schreiben
% ---------------------
  N = ['i' 'ndata'];
  dlmwrite(['protokol_linfit_' num2str(datum1) '.dat'], N,'delimiter', '\t'); 
% 
%  am unteren Rand variable Fitbreite
% ----------------------------------------
  for i = unten : unten + fit_breite
      ndata = (2*(i-unten)+1);
      % soll bei unten anfangen und nur nach oben hin fitten 
      % 
      % schreiben
          M = [alt(i),ndata, ndata*deltar];
          dlmwrite(['protokol_linfit_' num2str(datum1) '.dat'], M, 'delimiter', '\t', '-append'); 
      %  
      xx = [i-(i-unten):i+(i-unten)+1];
      x = range(i-(i-unten):i+(i-unten)+1); 
      y = input(i-(i-unten):i+(i-unten)+1); 
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x;   
    figure(1)
    box on 
    hold on
    plot(xx,y,'r')
    plot(xx,yy,'g')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata) - x(1));
  end
% ------------------------------
%      unten
% ------------------------------
  for i = unten + fit_breite+1 : mitte1 + fit_breite 
      ndata = 2*fit_breite+1;
      % schreiben
          M = [alt(i),ndata , ndata*deltar];
          dlmwrite(['protokol_linfit_' num2str(datum1) '.dat'], M, 'delimiter', '\t', '-append'); 
      % 
      xx = [i-fit_breite+1 : min(i+fit_breite+1,rbbb)];
      x = range(i-fit_breite+1 : min(i+fit_breite+1,rbbb));
      y = input(i-fit_breite+1 : min(i+fit_breite+1,rbbb));
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x; 
    plot(xx,y,'b')
    plot(xx,yy,'c')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata)- x(1));
        if i >= 0.8*rbbb
            return
        end
  end
% ------------------------------
%      mitte1
% ------------------------------
  for i = mitte1 + fit_breite+1 : mitte2 + fit_breite  
      ndata = 2*2*fit_breite+1;
      % schreiben
          M = [alt(i),ndata , ndata*deltar];
          dlmwrite(['protokol_linfit_' num2str(datum1) '.dat'], M, 'delimiter', '\t', '-append'); 
      % 
      xx = [i-2*fit_breite+1 : min(i+2*fit_breite+1,rbbb)];
      x = range(i-2*fit_breite+1 : min(i+2*fit_breite+1,rbbb));
      y = input(i-2*fit_breite+1 : min(i+2*fit_breite+1,rbbb));
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x; 
    plot(xx,y,'b')
    plot(xx,yy,'c')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata)- x(1));
       if i >= 0.8*rbbb
            i
            return
      end
  end  
% ------------------------------
%      mitte2
% ------------------------------
  for i = mitte2 + fit_breite+1 : oben + fit_breite  
      ndata = 2*4*fit_breite+1;
      % schreiben
          M = [alt(i),ndata , ndata*deltar];
          dlmwrite(['protokol_linfit_' num2str(datum1) '.dat'], M, 'delimiter', '\t', '-append'); 
      %  
      xx= [i-4*fit_breite+1 : min(i+4*fit_breite+1,rbbb)];
      x = range(i-4*fit_breite+1 : min(i+4*fit_breite+1,rbbb));
      y = input(i-4*fit_breite+1 : min(i+4*fit_breite+1,rbbb));
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt); 
    yy = a + b.*x; 
    plot(xx,y,'b')
    plot(xx,yy,'m')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata)- x(1));
        if i >= 0.8*rbbb
            return
        end
  end
  
% ------------------------------
%     oben
% ------------------------------
  for i = oben + fit_breite+1 : rbbb - 8*fit_breite-1 
      ndata = 2*8*fit_breite+1;
      % schreiben
          M = [alt(i),ndata , ndata*deltar];
          dlmwrite(['protokol_linfit_' num2str(datum1) '.dat'], M, 'delimiter', '\t', '-append'); 
      %   
      xx= [i-8*fit_breite+1 : i+8*fit_breite+1];
      x = range(i-8*fit_breite+1 : min(i+8*fit_breite+1,rbbb));
      y = input(i-8*fit_breite+1 : min(i+8*fit_breite+1,rbbb));
[a,b,siga,sigb,chi2,q] = fit_chi2(x,y,ndata,sig,mwt);
    yy = a + b.*x;
    plot(xx,y,'b')
    plot(xx,yy,'y')
    
 abl_chi2(i) = (yy(ndata) - yy(1))/...
                        (x(ndata)- x(1));
   if i >= 0.8*(rbbb - 8*fit_breite-1) 
       return
   end
  end
