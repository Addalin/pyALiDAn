% raman_beta_pollyXT.m
%
%   adapted for PollyXT_TROPOS in Haifa,  05/2019 BHEESE
%
% -------------------------------------------------------------
%  Raman Backscatter calculation: formula after Ansmann (1992)
% -------------------------------------------------------------
% beta_aero_355(z) = 
%            - beta_ray_355(z)+(beta_par(z0)+ beta_mol(z0)) ...
%           .*(P_387(z0).*P_355(z).*NR(z)/P_355(z0).*P_387(z0).*NR(z0) ...
%           .* exp(-int_z0_z(aero_ext_387 + ray_ext_387)/      
%              exp(-int_z0_z(aero_ext_355 + ray_ext_355)
%
% z0 is the aerosol free reference height -> beta_mol(z0) >> beta_par(z0) 
%                          -> beta_par(z0) + beta_par(z0) ... 
%                          =~ beta_mol(z0)
% no Overlap problem, since signal ratios are used.  
%
zet_0 = 1; 
%
clear p_ave_raman_1 p_ave_elast_1 p_ave_raman_2 p_ave_elast_2
clear m_ave_elast_1 m_ave_raman_1 m_ave_elast_2 m_ave_raman_2
clear exp_z_1 exp_n_1 exp_z_2 exp_n_2
clear xx yy 
clear beta_raman beta_raman_sm
clear Lidar_Ratio
%
up = 1500; 
%
Ref_1 = RefBin(1); % 355 nm
Ref_2 = RefBin(5); % 532 nm
Ref_3 = Ref_2;     % same as 532! 
%
exp_z_1(up)=1;
exp_n_1(up)=1;
exp_z_2(up)=1;
exp_n_2(up)=1;
exp_z_3(up)=1;
exp_n_3(up)=1;

% 
%  can be adjusted if too low 
% ----------------------------
beta_par(1,Ref_1)= 1e-12; %  355 
beta_par(2,Ref_2)= 1e-12; %  532     
beta_par(3,Ref_3)= 1e-12; % 1064
%

% *********
%  355 nm 
% *********
%for i=Ref_1 : -1 : zet_0 +1
   for i=up : -1 : zet_0 +1

%      for i=RefBin_1 : -1 : zet_0
% Raman Particle extinction at 387
   p_ave_raman_1(i) = 0.5*(aero_ext_raman(1,i) + aero_ext_raman(1,i-1))*aerosol_wellen_fak(1); 
% Raman molecular extinction at 387
   m_ave_raman_1(i) = 0.5*(ray_ext(3,i)+ray_ext(3,i-1));    
% Elastic particle  extinction at 355
   p_ave_elast_1(i) = 0.5*(aero_ext_raman(1,i) + aero_ext_raman(1,i-1)); 
% Elastic molecular extinction at 355
   m_ave_elast_1(i) = 0.5*(ray_ext(1,i)+ray_ext(1,i-1));
%       
   exp_z_1(i-1) = exp_z_1(i)* exp(-(p_ave_raman_1(i) + m_ave_raman_1(i))*deltar); 
   exp_n_1(i-1) = exp_n_1(i)* exp(-(p_ave_elast_1(i) + m_ave_elast_1(i))*deltar);
  end
%   
   for i=up : -1 : zet_0   
 signals_1(i) =(bg_corr_sm(3,Ref_1)*bg_corr_sm(1,i)*beta_mol(1,i))/...
    (bg_corr_sm(1,Ref_1)*bg_corr_sm(3,i)*beta_mol(1,Ref_1));  
 beta_raman(1,i)= - beta_mol(1,i)+ (beta_par(1,Ref_1)+ beta_mol(1,Ref_1))*signals_1(i)*exp_z_1(i)/exp_n_1(i);
   end
   
% *********  
%  532 nm 
% *********
 % for i=Ref_2 : -1 : zet_0 +1
for i=up : -1 : zet_0 +1
%      for i=RefBin_1 : -1 : zet_0
% Raman Particle extinction at 607
   p_ave_raman_2(i) = 0.5*(aero_ext_raman(2,i) + aero_ext_raman(2,i-1))*aerosol_wellen_fak(2); 
% Raman molecular extinction at 607
   m_ave_raman_2(i) = 0.5*(ray_ext(7,i)+ray_ext(7,i-1));    
% Elastic particle  extinction at 532
   p_ave_elast_2(i) = 0.5*(aero_ext_raman(2,i) + aero_ext_raman(2,i-1)); 
% Elastic molecular extinction at 532
   m_ave_elast_2(i) = 0.5*(ray_ext(5,i)+ray_ext(5,i-1));
%       
   exp_z_2(i-1) = exp_z_2(i)* exp(-(p_ave_raman_2(i) + m_ave_raman_2(i))*deltar); 
   exp_n_2(i-1) = exp_n_2(i)* exp(-(p_ave_elast_2(i) + m_ave_elast_2(i))*deltar);
  end
%   
   for i=up : -1 : zet_0   
 signals_2(i) =(bg_corr_sm(7,Ref_2)*bg_corr_sm(5,i)*beta_mol(5,i))/...
    (bg_corr_sm(5,Ref_2)*bg_corr_sm(7,i)*beta_mol(5,Ref_2));  
 beta_raman(2,i)= - beta_mol(5,i)+ (beta_par(5,Ref_2)+ beta_mol(5,Ref_2))*signals_2(i)*exp_z_2(i)/exp_n_2(i);
   end
%  
% *********  
%  1064 nm 
% *********
  for i=up : -1 : zet_0 +1
% Elastic particle extinction at 1064
   p_ave_elast_3(i) = 0.5*(aero_ext_raman(2,i) + aero_ext_raman(2,i-1))*aerosol_wellen_fak(3); 
% Elastic molecular extinction at 1064
   m_ave_elast_3(i) = 0.5*(ray_ext(8,i)+ray_ext(8,i-1));
%       
%   exp_z_3(i-1) = exp_z_3(i)* exp(-(p_ave_elast_2(i) + m_ave_elast_2(i) + ...
%        p_ave_raman_2(i) + m_ave_raman_2(i))*deltar); % 532 und 607
   exp_z_3(i-1) = exp_z_3(i)* exp(-(p_ave_raman_2(i) + m_ave_raman_2(i))*deltar); 
   exp_n_3(i-1) = exp_n_3(i)* exp(-2*(p_ave_elast_3(i) + m_ave_elast_3(i))*deltar);
  end
%   
   for i=up : -1 : zet_0   
 signals_3(i) =(bg_corr_sm(7,Ref_3)*bg_corr_sm(8,i)*beta_mol(8,i))/...
    (bg_corr_sm(8,Ref_3)*bg_corr_sm(7,i)*beta_mol(8,Ref_3));  
 beta_raman(3,i)= - beta_mol(8,i)+(beta_par(8,Ref_3)+ beta_mol(8,Ref_3))*signals_3(i)*exp_z_3(i)/exp_n_3(i);
   end
%  
% ------------
%  smoothing
% ------------
 beta_raman_sm(1,:) = smooth(beta_raman(1,:),21,'sgolay',3); 
 beta_raman_sm(2,:) = smooth(beta_raman(2,:),21,'sgolay',3);
 beta_raman_sm(3,:) = smooth(beta_raman(3,:),21,'sgolay',3);  
% 
% -------------
%  Lidar Ratio 
% -------------
Lidar_Ratio(1,:) = aero_ext_raman(1,zet_0:up)./beta_raman(1,zet_0:up);  
Lidar_Ratio(2,:) = aero_ext_raman(2,zet_0:up)./beta_raman(2,zet_0:up); 
%    
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%   Plots
% ------------
% Backscatter
% -------------
%rbbb = size(beta_aerosol);
rbbr = size(beta_raman_sm(1,:));
%
figure(15)
    set(gcf,'position',[50,100,600,800]); % units in pixels! *** 19 " ***
  % set(gcf,'position',[20,200,500,650]); % units in pixels! *** Laptop ***
  title(['Polly^X^T in Haifa on ' datum ', '  timex(i_start,1:5) ' - ' timex(i_stop,1:5) ' UTC '],'fontsize',[10]) 
  xlabel('Backscatter. coeff. / 1 / km sr','fontsize',[12])  
  ylabel('Height a.g.l. / m','fontsize',[12])
 % axis([-1e-3 5e-3 0 alt(Ref_1)]); 
axis([-0.2e-3 10e-3 0 alt(up)]); 
  box on
  hold on
 % plot(beta_aerosol(1,zet_0:rbbb(2)-1), alt(zet_0:rbbb(2)-1),'b--')
  plot(beta_raman_sm(1,zet_0:rbbr(2)), alt(zet_0:rbbr(2)),'b')
 % plot(beta_aerosol(4,zet_0:rbbb(2)-1), alt(zet_0:rbbb(2)-1),'g--')
  plot(beta_raman_sm(2,zet_0:rbbr(2)), alt(zet_0:rbbr(2)),'g')
 % plot(beta_aerosol(6,zet_0:rbbb(2)-1), alt(zet_0:rbbb(2)-1),'r--')
  plot(beta_raman_sm(3,zet_0:rbbr(2)), alt(zet_0:rbbr(2)),'r')
  betaref_1 =  num2str(beta_par(1,Ref_1), '%5.1e'); 
  betaref_2 =  num2str(beta_par(2,Ref_2), '%5.1e'); 
  betaref_3 =  num2str(beta_par(3,Ref_3), '%5.1e'); 
  refheight = [num2str(alt(Ref_1)*1e-3,'%5.1f') ' km'];
%  text(0.4*5e-3, 0.74*alt(Ref_1), ['Beta-Ref. 355 =' betaref_1 ' at ' refheight],'fontsize',10,'HorizontalAlignment','left')
%  text(0.4*5e-3, 0.70*alt(Ref_1), ['Beta-Ref. 532 =' betaref_2 ' at ' refheight],'fontsize',10,'HorizontalAlignment','left')
%  text(0.4*5e-3, 0.66*alt(Ref_1), ['Beta-Ref.1064 =' betaref_3 ' at ' refheight],'fontsize',10,'HorizontalAlignment','left')
  grid on
  legend('355','532','1064')
%  
% -------------- 
%  Lidar Ratio
% --------------
  rLR_1 = size(Lidar_Ratio(1,:));
  rLR_2 = size(Lidar_Ratio(2,:));
%  
  figure(16)
  %  set(gcf,'position',[50,100,600,800]); % units in pixels! *** 19 " ***
   set(gcf,'position',[20,200,500,650]); % units in pixels! *** Laptop ***
  
   title(['Polly^{XT} in Haifa on ' datum ', '  timex(i_start,1:5) ' - ' timex(i_stop,1:5) ' UTC '],'fontsize',[10]) 
  xlabel('Lidar Ratio / sr','fontsize',[12])  
  ylabel('Height a.g.l. / m','fontsize',[12])
%  axis([0 100 0 alt(Ref_1)]); 
  axis([0 100 0 alt(up)]); 
box on
  hold on
  plot(Lidar_Ratio(1,zet_0:rLR_1(2)),alt(zet_0:rLR_1(2)),'b')
  plot(Lidar_Ratio(2,zet_0:rLR_1(2)),alt(zet_0:rLR_1(2)),'g')
 
  grid on
  legend('355 nm', '532 nm')
  
  % 