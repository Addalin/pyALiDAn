%% read_sonde_pollyXT_ct80.m
%                                  01/07    BHeese 
%   Anpassung an RS80 SOnde        09/07    BHeese
%   Standardsonde justiert         11/07    BHeese
%   8 channels for PollyXT ct80    14/07    BHeese
%
clear altitude alti
clear beta_ray beta_par beta_mol pr2_ray_sig
clear temp pres relF 
% ------------------------------------------------------------
% Standard
% ------------------------------------------------------------
%welche Sonde
sonde = 0; 
%
rbins = 4000;
if sonde == 1   
    site = 'Standart';
    radiofile = 'c:\Polis\radiosonden\Standarts\AlbertStd.txt';
    disp(['*** Einlesen der Radiosonde ' radiofile]);
    sondedata = csvread(radiofile, 2, 0);
%
   altitude = sondedata(:,1)*1e3;  % in km
   beta_ray(1,:) = sondedata(:,4);
   beta_ray(4,:) = sondedata(:,4)./(4.259e-4).*(2.2686e-3);
   beta_ray(6,:) = sondedata(:,4)./(4.259e-4).*(2.582e-5);
%
nlines=201;
end 
% ---------------------------------------------
%  Radiosonde vom IFT Leipzig Type RS80
% ---------------------------------------------
if sonde == 2
    site = 'Leipzig';
    radiofile=['d:\Radiosonden\IfT\09052520.TXT'];
%     radiofile=['d:\Radiosonden\IfT\07092409.TXT'];
disp(['*** Einlesen der Radiosonde ' radiofile]);
fid=fopen(radiofile,'r');  
%
radios = [ '20' num2str(radiofile(20:27))]; 
%
   for j=1:39
    eval(['headerline_sonde' num2str(j) '=fgetl(fid);']);
   end   
%   
i=0;
 while ~feof(fid);
sondedata = fscanf(fid,'%g',6);
eval(['dummy' '=fgetl(fid);']); % NaN oder \\\\\
 if ~isempty(sondedata)
     content = size(sondedata);
     if content(1)<=3 
         continue  
     else
       i=i+1;
       pres(i) = sondedata(3);   % P in hPa!
   altitude(i) = sondedata(4);% in m!  %*1e-3 % in km
 %  if i>1 & altitude(i) < altitude(i-1) 
 %     altitude(i) = altitude(i-1)+0.0001;  
 %  end
       temp(i) = 273.16 + sondedata(5); % T in K
       relF(i) =  sondedata(6);
     end
%     if i > 1 ...
%   & altitude(i) < altitude(i-1) 
%     break
%   end   
% 
 else
   break 
 end % 
 end % while
%
% ---------------------
% radiosonde Wyoming 
% ---------------------
%yearstr = num2str(year);
elseif sonde == 0
%
%Lindenberg
%------------
% site = 'Lindenberg';
 site = 'Haifa';
 
 % id = '10393';
 id = '40179';
%
if hour(i_start)<=8
radiofile=['H:\data_haifa\Radiosonden\',site,'\',yearstr,'\',id,'_',datum2,'_00.dat'];  % Adi change the file location 
elseif hour(i_start) >=20
radiofile=['H:\data_haifa\Radiosonden\',site,'\',yearstr,'\',id,'_',datum3,'_00.dat'];
elseif hour(i_start) > 8 | hour(i_start) < 15
radiofile=['H:\data_haifa\Radiosonden\',site,'\',yearstr,'\',id,'_',datum2,'_12.dat'];
elseif hour(i_start) > 15 | hour(i_start) < 20
radiofile=['H:\data_haifa\Radiosonden\',site,'\',yearstr,'\',id,'_',datum2,'_18.dat'];
end 

%h = size(radiofile);
%hh = h(2);
%radios(1:8) = datum2; % num2str(radiofile(38:45)); 
%radios(9:10) = num2str(radiofile(47-4:47-5));  
radios = [datum2, num2str(radiofile(47-5:47-4))];  

%
disp(['*** Einlesen der Radiosonde ' radiofile]);
fid=fopen(radiofile,'r') ;
%
   for j=1:7
    eval(['headerline_sonde' num2str(j) '=fgetl(fid);']);
   end   
%   
i=0;
 while ~feof(fid)
 sondedata = fscanf(fid,'%g',11);
 if ~isempty(sondedata)
   i=i+1;
   pres(i)=sondedata(1);   % P in hPa!
   altitude(i)=sondedata(2); % in m 
   temp(i)=273.16 + sondedata(3); % T in K
   relF(i)=sondedata(5); 
 else
   break    
 end  
 end % while  
% 
end
if sonde ~= 1  
% Anzahl Sondenhöhen
 nlines = i;
%%
%*****************************************************
%        Beta berechnen
%*****************************************************
    beta_ray = zeros(8,nlines); 
   alpha_ray = zeros(8,nlines); 
    beta_mol = zeros(8,rbins); 
   alpha_mol = zeros(8,rbins); 
    beta_par = zeros(8,rbins); 
   alpha_par = zeros(8,rbins); 
  LidarRatio = zeros(8,rbins);
    BScRatio = zeros(8,1); 
 pr2_ray_sig = zeros(8,rbins);
%  
% ------------------------------------------
%  Rückstreuung berechnen  , temp in K  
% beta_ray - molecular
% ------------------------------------------
  beta_ray(1,:) = (2.265e-3).*pres./temp;  % 355 nm !!! Faktor in km!!!    - What is the left number meaning ? - is this Ts*Ns/Ns ? - Adi
  beta_ray(2,:) = (2.265e-3).*pres./temp;  % 355 nm 
  beta_ray(3,:) = (2.265e-3).*pres./temp*(355/387)^4.085;  % 387 nm Unsinn!? 
  beta_ray(4,:) = (2.265e-3).*pres./temp*(355/407)^4.085;  % 407 nm Unsinn!? 
  beta_ray(5,:) = (4.259e-4).*pres./temp;  % 532 nm
  beta_ray(6,:) = (4.259e-4).*pres./temp;  % 532 nm
  beta_ray(7,:) = (4.259e-4).*pres./temp*(532/607)^4.085;  % 607 nm Unsinn!?
  beta_ray(8,:) = (2.582e-5).*pres./temp;  % 1064 nm
end
% 
% ------------------------------------------------------
%  Lidar Ratio für Rayleigh-Streuung (8/3)pi = 8.377 sr
%  mit Depol Korrektur für Cabannes Linie
% -------------------------------------------------------
%  xlidar(1)=8.698; % 355 nm      S_c/(8/3)*pi = 1.0383
%  xlidar(2)=8.698; % 355 nm 
%  xlidar(3)=8.698; % ? 387 nm 
%  xlidar(4)=8.712; % 532 nm                   = 1.0400
%  xlidar(5)=8.712; % ? 607 nm  
%  xlidar(6)=8.736; % 1064 nm                  = 1.0426
% neu! 01/10 !!!   
  xlidar(1)=8.736; % 355 nm      S_c/(8/3)*pi = 1.0426
  xlidar(2)=8.736; % 355 nm 
  xlidar(3)=8.736; % ? 387 nm 
  xlidar(4)=8.736;  % 407 nm 
  xlidar(5)=8.712; % 532 nm                   = 1.0400
  xlidar(6)=8.712; % 532 nm                   = 1.0400
  xlidar(7)=8.712; % ? 607 nm  
  xlidar(8)=8.698; % 1064 nm                  = 1.0383
%   
% ----------------------
%  Rayleigh Extinktion
% alpha_ray - for mulecular
% ----------------------
alpha_ray (1,:) = beta_ray(1,:).*xlidar(1);
alpha_ray (2,:) = beta_ray(2,:).*xlidar(2);
alpha_ray (3,:) = beta_ray(3,:).*xlidar(3);
alpha_ray (4,:) = beta_ray(4,:).*xlidar(4);
alpha_ray (5,:) = beta_ray(5,:).*xlidar(5);
alpha_ray (6,:) = beta_ray(6,:).*xlidar(6);
alpha_ray (7,:) = beta_ray(7,:).*xlidar(7);
alpha_ray (8,:) = beta_ray(8,:).*xlidar(8);
  % 
% -------------------------------
%  Lidar Ratio für Aerosol 
% -------------------------------
   LidarRatio(1,:) = 55;
   LidarRatio(2,:) = 55;
   LidarRatio(3,:) = 55;
   LidarRatio(4,:) = 55; 
   LidarRatio(5,:) = 55;
   LidarRatio(6,:) = 55; 
   LidarRatio(7,:) = 55;
   LidarRatio(8,:) = 55; 
%
% -----------------------
%  Backscatter-Ratio ??? - 
% What is this ratio refer to ? (from the equations) why all values equal
% to 1 ? (Adi)
% -----------------------
   BScRatio(1)=1; %+0.35*LidarRatio(4)/LidarRatio(1)*(BScRatio(4)-1.); 
   BScRatio(2)=1; %+0.35*LidarRatio(4)/LidarRatio(1)*(BScRatio(4)-1.); 
   BScRatio(3)=1;
   BScRatio(4)=1;
   BScRatio(5)=1; % dann alpha_par und beta_par = 0 
   BScRatio(6)=1;
   BScRatio(7)=1;
   BScRatio(8)=1+6.00*LidarRatio(4)/LidarRatio(6)*(BScRatio(4)-1.); 
   
%   
% --------------------------------------
%  Interpolation auf Lidarstützstellen
% --------------------------------------
  beta_mol(1,:) = interp1(altitude(1:nlines),beta_ray(1,1:nlines),alt(1:rbins),'linear','extrap');
  beta_mol(2,:) = interp1(altitude(1:nlines),beta_ray(2,1:nlines),alt(1:rbins),'linear','extrap');
  beta_mol(3,:) = interp1(altitude(1:nlines),beta_ray(3,1:nlines),alt(1:rbins),'linear','extrap');
  beta_mol(4,:) = interp1(altitude(1:nlines),beta_ray(4,1:nlines),alt(1:rbins),'linear','extrap');
  beta_mol(5,:) = interp1(altitude(1:nlines),beta_ray(5,1:nlines),alt(1:rbins),'linear','extrap');
  beta_mol(6,:) = interp1(altitude(1:nlines),beta_ray(6,1:nlines),alt(1:rbins),'linear','extrap');
  beta_mol(7,:) = interp1(altitude(1:nlines),beta_ray(7,1:nlines),alt(1:rbins),'linear','extrap');
  beta_mol(8,:) = interp1(altitude(1:nlines),beta_ray(8,1:nlines),alt(1:rbins),'linear','extrap');

%% -------------
%  Rayleigh Signal 
% -----------------
  alpha_mol(1,:) = beta_mol(1,:).*xlidar(1); 
  alpha_mol(2,:) = beta_mol(2,:).*xlidar(2); 
  alpha_mol(3,:) = beta_mol(3,:).*xlidar(3); 
  alpha_mol(4,:) = beta_mol(4,:).*xlidar(4); 
  alpha_mol(5,:) = beta_mol(5,:).*xlidar(5); 
  alpha_mol(6,:) = beta_mol(6,:).*xlidar(6); 
  alpha_mol(7,:) = beta_mol(7,:).*xlidar(7); 
  alpha_mol(8,:) = beta_mol(8,:).*xlidar(8); 
  %  
% ------------
%  Particles 
% ------------
   beta_par(1,:) = beta_mol(1,:).*(BScRatio(1)-1.);  % WHAT BScRatio STANDS FOR ? -  ADI
   beta_par(2,:) = beta_mol(1,:).*(BScRatio(2)-1.);
   beta_par(3,:) = beta_mol(1,:).*(BScRatio(3)-1.);
   beta_par(4,:) = beta_mol(4,:).*(BScRatio(4)-1.);
   beta_par(5,:) = beta_mol(5,:).*(BScRatio(5)-1.);
   beta_par(6,:) = beta_mol(6,:).*(BScRatio(6)-1.);
   beta_par(7,:) = beta_mol(7,:).*(BScRatio(7)-1.);
   beta_par(8,:) = beta_mol(8,:).*(BScRatio(8)-1.);
   
  alpha_par(1,:) = beta_par(1,:).*LidarRatio(1); 
  alpha_par(2,:) = beta_par(2,:).*LidarRatio(2); 
  alpha_par(3,:) = beta_par(3,:).*LidarRatio(3); 
  alpha_par(4,:) = beta_par(4,:).*LidarRatio(4); 
  alpha_par(5,:) = beta_par(5,:).*LidarRatio(5); 
  alpha_par(6,:) = beta_par(6,:).*LidarRatio(6); 
  alpha_par(7,:) = beta_par(7,:).*LidarRatio(7); 
  alpha_par(8,:) = beta_par(8,:).*LidarRatio(8); 
% 
tau=0;
r_bin = (range(2)-range(1))*1e-3;
for j = 1:8
  for i=1:rbins
      alpha_par(j,i)=0;
      tau = tau + alpha_mol(j,i)*r_bin; 
      zet2(i)= i*r_bin*i*r_bin; 
      ray_signal(j,i)=(1./zet2(i)).*(beta_mol(j,i)*exp(-2.*tau)); 
      pr2_ray_sig(j,i)= ray_signal(j,i)*zet2(i); 
  end
end
%

figure(10)
  set(gcf,'position',[50,100,600,800]); % units in pixels! *** 19 " ***
%  set(gcf,'position',[20,200,400,650]); % units in pixels! *** Laptop ***
% 
if sonde ~= 1
title(['Radiosounding at ' site ' on ' datum2 ' at ' radios(9:10) ' UTC' ],'fontsize',[14]) 
end

%%
%  Plot
% -------
% ADI - WHAT ARE  THE 8 CHANNELS ? Whay using only 5 of them ?
hl1=line(beta_mol(1,:),alt(1:rbins)*1e-3,'Color','b');  % 355 nm  uv
%hl2=line(beta_mol(2,:),alt(1:rbins)*1e-3,'Color','b'); 
hl3=line(beta_mol(3,:),alt(1:rbins)*1e-3,'Color','c');  % 387 nm  blue   visable: 380nm--780nm
hl4=line(beta_mol(5,:),alt(1:rbins)*1e-3,'Color','g'); % 532 nm   green
hl5=line(beta_mol(7,:),alt(1:rbins)*1e-3,'Color','r');  % 607 nm   red
hl6=line(beta_mol(8,:),alt(1:rbins)*1e-3,'Color','m'); % 1064 nm  ir
%hl4=line(beta_ray(1,:),altitude(:)*1e-3,'Color','b');
%hl5=line(beta_ray(4,:),altitude(:)*1e-3,'Color','g');
%hl6=line(beta_ray(6,:),altitude(:)*1e-3,'Color','r');
hold on
ax1 = gca;

xlimits = get(ax1,'XLim');
ylimits = get(ax1,'YLim');
xinc = (xlimits(2)-xlimits(1))/8;
yinc = (ylimits(2)-ylimits(1))/5;

set(ax1,'XTick',[xlimits(1):xinc:xlimits(2)],...
        'YTick',[ylimits(1):yinc:ylimits(2)]);
    
    ylabel(ax1,'Altitude / m')
    xlabel(ax1,'Lidar Beta / m-1')
    grid on
    
 legend('355','387','532','607','1064');

%else
% 

if sonde ~= 1
figure(11)
  set(gcf,'position',[50,100,600,800]); % units in pixels! *** 19 " ***
%  set(gcf,'position',[20,200,400,650]); % units in pixels! *** Laptop ***

title(['Radiosounding at ' site ' on ' radios(1:8) ' at ' radios(9:10) ' UTC'],'fontsize',[14]) 
hl1=line(temp,altitude,'Color','r');
hold on
ax1 = gca;
set(ax1,'XColor','r','YColor','k')

ax2 = axes('Position',get(ax1,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','b','YColor','k');
       
hl2 = line(relF,altitude,'Color','b','Parent',ax2);

xlimits = get(ax1,'XLim');
ylimits = get(ax1,'YLim');
xinc = (xlimits(2)-xlimits(1))/5;
yinc = (ylimits(2)-ylimits(1))/5;

set(ax1,'XTick',[xlimits(1):xinc:xlimits(2)],...
        'YTick',[ylimits(1):yinc:ylimits(2)]);
    
    ylabel(ax1,'Altitude / m')
    xlabel(ax1,'Temperature / K')
    xlabel(ax2,'rel Humidity / %')

    grid on

% legend('Temperature', 'rel Feuchte');%, 'Wind stärke', 'Windrichtung');
%
end 
%
  disp('End of program: read_sonde_pollyXT_ct80.m, Vers. 1.0 07/14')