% klett_pollyXT.m
%
% aus Klett_1profile.m       01/07 BHeese 
% Anpassungen                10/07 BHe
% nach untenund oben!        11/08 BHe
%
% vorher folgende M-Files laufen lassen:  
%
%       read_pollyXT.m
%       read_sonde_polly.m
%       rayleigh_fit_PollyXT.m
%
clear beta_par alpha_par
clear rc_signal
clear ext_ray
clear ext_par
clear fkt1
clear fkt2
clear zfkt
clear nfunk
clear extinktion
clear beta_aero beta_bv
clear beta_aerosol beta_aerosol_sm
clear alpha_aerosol alpha_aerosol_sm
clear alpha_aero
% 
zet_0 = 2;      
sm_span = 11;  % Spannweite für Savitzky_Golay smoothing  * 30 m 
rbins = 3000; 
%
%  höhenabhängiges LidarRatio setzten
% ---------------------------------------
% 
% lower_layer1=round(6.3/7.5e-3) 
% upper_layer1=round(8.0/7.5e-3)
%   
% lower_layer = 840;
% upper_layer = 900;
%
% for i=lower_layer1:upper_layer1
% LidarRatio_532(i) = 55;  % Leipzig
% end
%
% second layer
%
% lower_layer = 901;
% upper_layer = 980;
%
% for i=lower_layer1:upper_layer1
% LidarRatio_532(i) = 55;  % Leipzig
% end
%
% **************************************************
%  Referenzwerte für alpha und beta Partikel setzen
% **************************************************
 beta_par(1,RefBin(1)) = 1e-12;
 beta_par(5,RefBin(5)) = 1e-12;
 beta_par(8,RefBin(8)) = 1e-12;
%
 alpha_par(1,RefBin(1)) = beta_par(1,RefBin(1))*LidarRatio(1,RefBin(1)); 
 alpha_par(5,RefBin(5)) = beta_par(5,RefBin(5))*LidarRatio(5,RefBin(5)); 
 alpha_par(8,RefBin(8)) = beta_par(8,RefBin(8))*LidarRatio(8,RefBin(8)); 
%
% im Cirrus anpassen  
% -------------------
% Ref_Bin = 399;
% Ref_Bin = 403; % 23.06.09
% alt(Ref_Bin)
% beta_par(1,Ref_Bin) = 2.8e-3;
% beta_par(4,Ref_Bin) = 2.8e-3;
% beta_par(6,Ref_Bin) = 2.8e-3;

%Ref_Bin = 398; % 25.05.09 21 - 23h IfT
%alt(Ref_Bin)
% beta_par(1,Ref_Bin) = 7e-3;
% beta_par(4,Ref_Bin) = 7e-3;
% beta_par(6,Ref_Bin) = 7e-3;
%
% alpha_par(1,Ref_Bin) = beta_par(1,Ref_Bin)*LidarRatio(1,RefBin(1)); 
% alpha_par(4,Ref_Bin) = beta_par(4,Ref_Bin)*LidarRatio(4,RefBin(4)); 
% alpha_par(6,Ref_Bin) = beta_par(6,Ref_Bin)*LidarRatio(6,RefBin(6)); 
%
 for j=1:8 % & Kanäle
        if j==1 | j==5 | j==8 
% ----------------------------------    
%         xlidar(j,:) = xlidar(j);          
% ---------------------                 
%  Signal aussuchen
% ---------------------
  rc_signal(j,:) = pr2_sm(j,:);  % SG smoothed über 330m! 
%  Ref_Bin = min(RefBin); 
  Ref_Bin = RefBin(j); 
  ext_ray(j,:) = alpha_mol(j,:); 
%
% ***********************************
%  Rückstreukoeffizient: beta_par
% ***********************************
%   
  beta_par(j,Ref_Bin-1) = beta_par(j,Ref_Bin); 
  beta_bv(j) = beta_par(j,Ref_Bin)+ beta_mol(j,Ref_Bin);
% leere Kanäle  
  if beta_bv(j) == 0
      beta_bv(j) = 1;
  end 
%  
% -------------------------------------------------------------------------
%  Klett (Gl. 20; 1985):
%  fkt1: 2/B_R int(beta_R) = 2 int (alpha_R) = sum (alpha_1,R+alpha_2,R)*deltar
%  fkt2: 2 int(beta_R/B_P) = 2 int(alpha_R*B_R/B_P) =  2 int (alpha_R*S_P/S_R)
% -------------------------------------------------------------------------
%  Fernald, AO, 1984
%
ext_ave =(ext_ray(j,Ref_Bin) + ext_ray(j,Ref_Bin-1)) * deltar;
fkt1(Ref_Bin) = ext_ave; 
fkt2(Ref_Bin) = ext_ave/xlidar(j) * LidarRatio(j,Ref_Bin); 
% 
  for i=Ref_Bin-1 : -1 : zet_0
   ext_ave = (ext_ray(j,i) + ext_ray(j,i-1)) * deltar; 
   fkt1(i) = fkt1(i+1) + ext_ave; 
   fkt2(i) = fkt2(i+1) + ext_ave/xlidar(j) * LidarRatio(j,i); 
  end
% 
% -------------------------------------------------------------
%  zfkt: exp(S'-Sm') laut Klett (Gl 22)(Paper 1985) = S-Sm+fkt1-fkt2 
% ----------------------------------------------------------------
  for i=zet_0:Ref_Bin
    zfkt(i)=rc_signal(j,i)/rc_signal(j,Ref_Bin)/exp(fkt1(i))*exp(fkt2(i));
  end
%
% Integral im Nenner (2. Summand); 2 kürzt sich mit Mittelwert-Halbe

   nfkt(Ref_Bin)=zfkt(Ref_Bin)*deltar/LidarRatio(j,Ref_Bin); 
     for i=Ref_Bin-1: -1 : zet_0
       nfkt(i)=nfkt(i+1)+(zfkt(i)+zfkt(i+1))*deltar*LidarRatio(j,i); 
     end
% 
% Klett 1985, Gl. (22)
%
  for i=Ref_Bin-1 : -1 : zet_0+1
  beta_aero(j,i) = zfkt(i)/(1./beta_bv(j) + nfkt(i)); 
  end
%  
%  +++++++++++++++++++++++
%    nochmal nach oben 
%  +++++++++++++++++++++++
     for i=Ref_Bin : rbins-1
   ext_ave = (ext_ray(j,i) + ext_ray(j,i+1)) * deltar; 
   fkt1(i) = fkt1(i-1) + ext_ave; 
   fkt2(i) = fkt2(i-1) + ext_ave/xlidar(j) * LidarRatio(j,i); 
  end
% 
% -------------------------------------------------------------
%  zfkt: exp(S'-Sm') laut Klett (Gl 22)(Paper 1985) = S-Sm+fkt1-fkt2 
% ----------------------------------------------------------------
  for i=Ref_Bin : rbins-1
    zfkt(i)=rc_signal(j,i)/rc_signal(j,Ref_Bin)/exp(fkt1(i))*exp(fkt2(i));
  end
%
% Integral im Nenner (2. Summand); 2 kürzt sich mit Mittelwert-Halbe

   nfkt(Ref_Bin)=zfkt(Ref_Bin)*deltar/LidarRatio(j,Ref_Bin); 
     for i=Ref_Bin : rbins-1
       nfkt(i)=nfkt(i-1)+(zfkt(i)+zfkt(i-1))*deltar*LidarRatio(j,i); 
     end
% 
% Klett 1985, Gl. (22)
%
  for i=Ref_Bin : rbins-1
  beta_aero(j,i) = zfkt(i)/(1./beta_bv(j) + nfkt(i)); 
  end
%  ++++++++++++++++++++++++
%
% Rückstreu Profil 
%for i=1:Ref_Bin-1
   for i=1:rbins-1
    if i <= zet_0
      beta_aerosol(j,i) = NaN; 
     alpha_aerosol(j,i) = NaN; 
    else
 % für beta_Aerosol muß das beta_mol abgezogen werden 
     beta_aerosol(j,i) = beta_aero(j,i) - beta_mol(j,i); 
    alpha_aerosol(j,i) = beta_aerosol(j,i) * LidarRatio(j,i); 
   end
end   
%********************************
   end
end %  Anzahl Wellenlängen j
%********************************
%-------------
%  glätten 
%-------------
for j=1:8
     if j==1 | j==5 | j==8 
 beta_aerosol_sm(j,:) = smooth(beta_aerosol(j,:),sm_span,'sgolay',3);
alpha_aerosol_sm(j,:) = smooth(alpha_aerosol(j,:),sm_span,'sgolay',3);
     end
end 
%----------
%  Plots
%----------
rb = Ref_Bin; 
%--------------
% Beta Aerosol
%--------------
scrsz = get(0,'ScreenSize'); 
figure(13)
set(gcf,'position',[scrsz(3)-0.95*scrsz(3) scrsz(4)-0.95*scrsz(4) scrsz(3)-0.4*scrsz(3) scrsz(4)-0.15*scrsz(4)]);  
%
  subplot(1,2,1)
  title(['Polly^X^T Aerosol Backscatter'],'fontsize',[14]) 
  xlabel('Backsc. coeff. / km-1','fontsize',[12])  
  ylabel('Height a.s.l. / km','fontsize',[12])
  axis([-1e-3 11e-3 0 alt(1,rbins)]); 
 axis([min(beta_aerosol(1,100:rbins/2)) max(beta_aerosol(1,1:rbins/2))+0.1*max(beta_aerosol(1,1:rbins/2)) 0 alt(1,rbins/2)]); 
  box on
  hold on
 plot(beta_aerosol(1,1:rbins-1), alt(1,1:rbins-1),'b'); 
 plot(beta_aerosol(5,1:rbins-1), alt(1,1:rbins-1),'g'); 
 plot(beta_aerosol(8,1:rbins-1), alt(1,1:rbins-1),'r'); 
 %plot(beta_mol(4,1:rb-1), alt(1,1:rb-1)); 
 legend('355 nm','532 nm','1064 nm'); 
 grid on 
 %
 %annotation('textbox', [0.62 0.8 0.28 0.04]);
 %text(0.1*10e-3, alt(1,rb)-0.02*alt(1,rb),...
 %   {[hourx(1,:) ':' minutex(1,:) ':' secondx(1,:) ' - '...
 %   hourx(nmeas,:) ':' minutex(nmeas,:) ':' secondx(nmeas,:) ' UTC ']} ,'FontSize',[10]);
%
% ----------------
%  Alpha Aerosol
% ----------------
subplot(1,2,2)
 title([datum ', ' timex(i_start,1:5) ' - ' timex(i_stop,1:5) ' UTC'],'fontsize',[14]) 
  xlabel('   Ext. coeff. / km-1','fontsize',[12])   
%axis([-0.05 0.5 0 alt(1,rbins)]); 
axis([min(alpha_aerosol(1,100:rbins/2)) max(alpha_aerosol(1,1:rbins/2))+0.1*max(alpha_aerosol(1,1:rbins/2)) 0 alt(1,rbins/2)]);
  box on
  hold on
 plot(alpha_aerosol(1,1:rbins-1), alt(1,1:rbins-1),'b');
 plot(alpha_aerosol(5,1:rbins-1), alt(1,1:rbins-1),'g');
 plot(alpha_aerosol(8,1:rbins-1), alt(1,1:rbins-1),'r');
 legend('355 nm','532 nm','1064 nm'); 
 grid on
% 
% 