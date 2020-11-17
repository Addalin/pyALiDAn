% Raman_polly.m
% 
%  Berechnung des Aerosol Extinktionskoeffizienten 
%  aus Polly Raman Messungen 
% 
%  03.04.06 BHeese
%  30.07.06 BHeese
%  nach erfolgreicher Raman Testauswertung (Gelsonmina 2005) angepasst  13.10.06 BHeese 
%  08/07  für PollyXT angepaßt  BHeese
%  for Polly XT UWA NF  02/15   BHeese
% 
%  vorher folgende M-Files laufen lassen:  
%
%         read_pollyXT.m
%   read_sonde_polly.m
% Rayleigh_fit_pollyXT.m
%        klett_pollyXT.m
% 
clear log_raman
clear abl_Raman
clear aero_ext_raman
clear aero_ext_raman_sm
clear signal
clear ray_ext
%
% -----------------------
%   Rayleigh Extinktion 
% -----------------------
% sig_ray aus Rayleigh_fit_pollyt.m
%   
% Rayleigh Wellenlängenabhängigkeit 
% ----------------------------------
  ray_fak(1) = (355/387)^4.085; 
  ray_fak(2) = (532/607)^4.085; 
  ray_fak(3) = (532/1064)^4.085; 
% 
% Angström Koeffizient 
% -------------------------
aang = 1.05; % Leipzig
% aang = 0.2;     % Wüstenaerosol   
 aerosol_wellen_fak(1) = (355/387)^aang;  
 aerosol_wellen_fak(2) = (532/607)^aang;  
 aerosol_wellen_fak(3) = (532/1064)^aang; 
% 
  ray_ext(1,:) = alpha_mol(1,:);   % 355 nm
  ray_ext(3,:) = ray_ext(1,:).*ray_fak(1);   % 387 nm
  ray_ext(5,:) = alpha_mol(5,:);  % 532 nm
  ray_ext(7,:) = ray_ext(5,:).*ray_fak(2);  % 607nm
  ray_ext(8,:) = ray_ext(5,:).*ray_fak(3);   %1064nm 
  
% ----------------
  for i = 1:rbins  
% ----------------  
%   Logarithmus
% ---------------- 
  log_raman(3,i) = log(ray_ext(3,i)/pr2_sm(3,i)); % 387 nm
  log_raman(7,i) = log(ray_ext(7,i)/pr2_sm(7,i)); % 607 nm
%
  end 
% --------------
%    Ableitung
% --------------    % 5-10 am abends, 10-20 tags
  fit_breite = 10;   % 1)
                    % bei ableitung_chi2_increasing.m
                    % progressive fitbreite mit der Höhe zunehmend: 
                    % am Anfang unten bis unten + fit_breite+1 ableiten 
                    % über ndata = (2*(i-unten)+1) = 3,5,7,9,11,...
                    % danach immer in 2*fit_breite Schritten 2 rbins mehr! 
                    % bei fit_breite = 5: in 1 km ~500 m, in 2 km ~700 m, 
                    % in 3km ~900 m, in 4 km ~ 1100m, in 5 km ~ 1300m 
                    % 2)
                    % bei ableitung_chi2_doubling.m, (Poisson Statistik) 
                    % verdoppelt sich die feste fitbreite (5, 10 oder 20
                    % rbins) bei den Punkten "mitte 1", "mitte 2" und  "oben"
                    % 3)
                    % in ableitung_chi2_fest.m ansteigend bis
                    % unten + fit_breite, danach fest 
                    %
% -------
% * 387 *  
% -------
% 1)
 [abl_chi2_1,chi2,q,u] = ableitung_chi2_increasing(log_raman(3,:),range./1e3,rbins-1,fit_breite,datum1);
% 2)
% [abl_chi2_1,chi2,q,u] = ableitung_chi2_doubling(log_raman(3,:),range./1e3,alt./1e3,rbins-1,fit_breite,datum1);
% 3)
% [abl_chi2_1,chi2,q,u] = ableitung_chi2_fest(log_raman(3,:),range./1e3,alt./1e3,rbins-1,fit_breite,datum1);
 abl_Raman(1,:) = abl_chi2_1; 
 rb2_1 = size(abl_Raman(1,:));
% -------
% * 607 *
% -------
% 1)
  [abl_chi2_2,chi2,q,u] = ableitung_chi2_increasing(log_raman(7,:),range./1e3,rbins-1,fit_breite,datum1);
% 2)
% [abl_chi2_2,chi2,q,u] = ableitung_chi2_doubling(log_raman(7,:),range./1e3,alt./1e3,rbins-1,fit_breite,datum1);
% 3)
% [abl_chi2_2,chi2,q,u] = ableitung_chi2_fest(log_raman(7,:),range./1e3,alt./1e3,rbins-1,fit_breite,datum1);
% 
abl_Raman(2,:) = abl_chi2_2; 
 rb2_2 = size(abl_Raman(2,:));
%
rb3 = max(rb2_1(2), rb2_1(2));
% 
% 
% ---------------------
%   Raman Extinktion
% ---------------------
aero_ext_raman = NaN(2,rb3);
%
for i=u:rb3 % p.11 L.4
 aero_ext_raman(1,i) = (abl_Raman(1,i)-ray_ext(1,i)-ray_ext(3,i))./(1+aerosol_wellen_fak(1));
 aero_ext_raman(2,i) = (abl_Raman(2,i)-ray_ext(5,i)-ray_ext(5,i))./(1+aerosol_wellen_fak(2));
end
%
% --------
%   Plot
% --------
rbb_a = rb3; 
rbb_p = RefBin(3); 
rbb_ka = RefBin(2); 
rbb_kp = RefBin(3); 
zet_0 = 21;
%
figure(12) 
  set(gcf,'position',[50,100,600,800]); % units in pixels! *** 19 " ***
% set(gcf,'position',[20,200,500,650]); % units in pixels! *** Laptop ***
 title(['Polly XT IfT on ' datum ', ' timex(i_start,1:5) ' - ' timex(i_stop,1:5) ' UTC '],'fontsize',[10]) 
  xlabel('Extinction 1/km','fontsize',[10])  
  ylabel('height a.s.l. / m','fontsize',[10])
   axis([-0.1 0.5 alt(zet_0) alt(rbb_ka)]); 
  box on 
  hold on 
 % Raman 
 plot(aero_ext_raman(1,zet_0:rbb_a),alt(zet_0:rbb_a),'b')
 plot(aero_ext_raman(2,zet_0:rbb_a),alt(zet_0:rbb_a),'g')
 % Klett
% plot(alpha_aerosol(1,zet_0:rb-1),alt(zet_0:rb-1),'b--')
% plot(alpha_aerosol(4,zet_0:rb-1),alt(zet_0:rb-1),'g--')
 %
 legend('Raman 355','Raman 532', 'Klett 355', 'Klett 532')
 grid on
%
%  end of program

