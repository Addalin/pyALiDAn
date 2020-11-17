% read_pollyXT_ct80.m
%
%  BHeese, 07/14    
%
% reads Polly XT ct80 raw data from Netcdf Format
% pretrigger of 248 rangebins
%
% -------------------
%  open netcdf files
% -------------------
%
clear 
close all
%
if exist('ncdfpath','var')
else
ncdfpath = 'C:\data_haifa\';
end
%
[ncdfname,ncdfpath,FilterIndex] = uigetfile([ncdfpath '*.nc']);
%
% for i = 1:nfiles % more than one file? 
% [ncid, status] = mexnc ( 'open',[ncdfpath ncdfname_c(i,:)]);
ncid = netcdf.open([ncdfpath ncdfname]);
disp(['open ' ncdfpath ncdfname])
% end
%  
[ndims, nvars, ngatts, unlimdimID]= netcdf.inq(ncid);
%          Inquires as to the number of dimensions, number of variables,
%          number of global attributes, and the unlimited dimension (if
%          any).  
%
 for i=1:ndims
 [name, length] = netcdf.inqDim(ncid,i-1); 
 cell_name(i,:) = cellstr(name);   % Fields names:   'time' , 'height' ,  'channel' ,  'date_time',  'coordinates', 'polynomial'
 cell_length(i,:) = length;             % Dimensions:       720 time stamps, 6400 bins (hight bins), 13 channels, 2 (date & time ? ), 2 coordinate (Latitude & longitude ?),  polynomial (?)
 
  end   
       nmeas = cell_length(1);
   rangebins = cell_length(2); 
       nchan = cell_length(3); 
         ndt = cell_length(4); 
       ncoor = cell_length(5); 
 % 
 clear name length

% allocate 
 datatypes = zeros(nvars,1);
  num_dims = zeros(nvars,1);
    dim_id = zeros(nvars,1); 
   num_att = zeros(nvars,1);
   
   for i=1:nvars
     
 [varname, datatype, natts] = netcdf.inqVar(ncid, i-1);
 cell_varname(i,:) = cellstr(varname);
 datatypes(i) = datatype;
% num_dims(i) = ndims;
% dim_id(i) = dimids;
% num_att(i) = natts;
 clear varname datatype ndims dimids natts 
 end
 
 raw_signal = netcdf.getVar(ncid, 0);
 measurement_shots  = netcdf.getVar(ncid, 1);  %What 9 measurments shots ? 
 measurement_time = netcdf.getVar(ncid, 2);
 depol_cal_angle = netcdf.getVar(ncid, 3);
 measurement_height_resolution = netcdf.getVar(ncid, 4 );  % tau
 laser_rep_rate = netcdf.getVar(ncid, 5);
 laser_power = netcdf.getVar(ncid, 6);
 laser_flashlamp = netcdf.getVar(ncid, 7);
 location_height = netcdf.getVar(ncid, 8);
 location_coordinates  = netcdf.getVar(ncid, 9);
 neutral_density_filter  = netcdf.getVar(ncid, 10);
 if_center = netcdf.getVar(ncid, 11);
 if_fwhm  = netcdf.getVar(ncid, 12);
 polstate=netcdf.getVar(ncid, 13);%		polstate:long_name = "Polarization state of the channels. 0=total, 1=co, 2=cross." ;
 telescope=netcdf.getVar(ncid, 14);%		telescope:long_name = "Telescope for each channel. 0=far range, 1=near range." ;
 deadtime_polynomial=netcdf.getVar(ncid, 15);%		deadtime_polynomial:long_name = "5-order polynomial coefficients for dead time correction at MCPS scale" ;
 deadtime_polynomial_error=netcdf.getVar(ncid, 16); %		deadtime_polynomial_error:long_name = "error of dead time polynomial coefficients" ;
 discr_level  = netcdf.getVar(ncid, 17);
 pm_voltage  = netcdf.getVar(ncid, 18);
 pinhole = netcdf.getVar(ncid, 19);
 zenithangle = netcdf.getVar(ncid, 20);
% 
%------------------------
% Global attributes
%------------------------%
% [att_value, status] = mexnc ( 'get_att_XXX', ncid, varid, attname );
%   Possibilities for XXX include 
%   "uchar", "schar", "short", "int", "float", and "double".  
%   The data is automatically converted to the external type 
%   of the specified attribute.   
      
location = netcdf.getAtt(ncid,netcdf.getConstant('NC_GLOBAL'), 'location');
comment = netcdf.getAtt(ncid,netcdf.getConstant('NC_GLOBAL'), 'measurement_comment');
%
       netcdf.close(ncid);
%            Closes a previously-opened NetCDF file.
%
% ---------------------------
%  calculate Measuring times
%  E.g. : times stamps are for 6 hours, every 30 sec (120 ,measurments in an hour ) , total of 6[hrs]X120[mes/hr]=720
% ---------------------------
 date1 = measurement_time(1,:); 
 time = measurement_time(2,:); 
 nmeas = size(time);
% 
 minutes = rem(time,3600); 
 hour = (time-minutes)/3600; 
 second = rem(minutes,60); 
 minute = (minutes-second)/60;
% 
% Choose measurement period?  
choose = true;
%choose = true; 
%start = [00,11,0];
%stop  = [23,59,0];
start = [06,00,00]
stop = [11,59,00]
% start = [13,12,00];
% stop  = [13,14,00];
%
if choose
for i=1:nmeas(2)
     if (hour(i) == start(1)) && (minute(i) == start(2)) %& (second(i) == start(3))
         i_start = i;
     end    
     if (hour(i) == stop(1)) && (minute(i) == stop(2))  %& (second(i) == stop(3))
         i_stop = i;
     end   
end  
else
i_start = 1;
i_stop = nmeas(2);
    end 
%
% ------------------------------------------
% Time -> String conversion 
% ------------------------------------------
for i = 1:nmeas(2)
%    
 if hour(i) < 10 
          hourx(i,:) = ['0' num2str(hour(i))];
      else     
          hourx(i,:) = num2str(hour(i));
 end    
%
 if minute(i) < 10
          minutex(i,:) = ['0' num2str(minute(i))];
      else     
          minutex(i,:) = num2str(minute(i));
 end    
% 
 if second(i) < 10
          secondx(i,:) = ['0' num2str(second(i))];
      else     
          secondx(i,:) = num2str(second(i));
 end    
%
 timex(i,:) = [hourx(i,:) ':' minutex(i,:) ':' secondx(i,:)];            
end
% ------------------------------------------
% Datum für die Plots als String umwandeln 
% ------------------------------------------      
    datum1 = num2str(date1(i_start)); 
     datum = [datum1(7:8) datum1(5:6)  datum1(1:4)];
    datum2 = [datum1(1:4) datum1(5:6) datum1(7:8)];
    datum3 = num2str(date1(1)+1);
   yearstr = datum2(1:4);
%     
% ---------------------------
%  Höhenaufloesung z=c*t/2  
% ---------------------------
c = 299792458; % m/s
% ---------------------------
% topographische Höhe
%----------------------------
% height = location_height;
height = 250; % Haifa   
% height = 117; % Leipzig   	
% height = 87;  % Melpitz	DE	51,533 	12,900 
% height = 129; % Manaus	BR	-59,961	-2,591 
% height = 260; % Gual Pahari IN   2,4  - 77,0
%----------------------------
% first Rangebin is shorter
%----------------------------
% pretrigger 
pt = 248; 
% 

raw_bg(:,1:pt,:) = raw_signal(:,1:pt,:);  % The dimentions of raw_signal : 13 channels X 6400 hight bins X 720 times stamps  
signal(:,1:rangebins-pt,:) = raw_signal(:,pt+1:rangebins,:);   % The signal without pre-triggered : 13 channels X 6152 (249:6400) hight bins X 720 times stamps  
sum_raw_signal = sum(raw_signal(:,1:rangebins-pt,i_start:i_stop),3);  % shouldn't is be signal ? or start from pt (not from 1)

sum_raw_bg     = sum(raw_bg(:,1:pt,i_start:i_stop),3);
sum_signal     = sum(signal(:,1:rangebins-pt,i_start:i_stop),3);
sum_shots      = sum(measurement_shots,2); 
%
for i= 1:rangebins-pt 
%alt(i) = height + sin((90-zenithangle)*(pi/180))* i * (c*measurement_height_resolution*1e-9)/2; 
range(i) = i * (c*measurement_height_resolution*1e-9)/2;  % dt = 50 [nsec]
alt(i) = height +  range(i); 
end 
%
deltar =(c*measurement_height_resolution*1e-9)/2*1e-3; % 0.0075 [km] ~ 7.5[m]
%
for j=1:nchan
mean_raw_signal(j,:)  = sum_raw_signal(j,:)./sum_shots(j); 
mean_signal(j,:)      = sum_signal(j,:)./sum_shots(j); 
mean_raw_bg(j,:)      = sum_raw_bg(j,:)./sum_shots(j); 
%total_raw_signal(j,:) = sum_signal(j,:); 
end 
%
%-----------------------
% Überlapp korrigieren
%-----------------------
%overlap_name(1,:) = ['pollyxt_ift-06082007-2020utc.o34'];
%overlap_name(1,:) = ['pollyxt_ift-23092007-0020utc-355.o34'];
%overlap_name(2,:) = ['pollyxt_ift-23092007-0020utc-532.o34'];
%overlap_name(1,:) = ['pollyxt-fmi-20032008-2200utc-355.o34'];
%overlap_name(2,:) = ['pollyxt-fmi-20032008-2200utc-532.o34'];
%overlap_name(1,:) = ['pollyxt_ift-24092007-01-03utc-532.o34'];
%overlap_name = ['pollyxt_ift-24092007-01-03utc-532.o34'];
% 11/2009
%overlap_name(1,:) = ['pollyxt_ift-20090509-2230-2330utc-40sr-355-3km-complete.o34'];
%overlap_name(2,:) = ['pollyxt_ift-20090509-2230-2330utc-50sr-532-3km-complete.o34'];
%overlap_name(1,:) = pollyxt_ift-20090113-2000utc-532-LE.o34'];
%overlap_name(2,:) = ['pollyxt_ift-20090113-2000utc-532-LE.o34'];
%[mean_raw_signal]=overlap_pollyXT(mean_raw_signal,overlap_name);  
%
% 0.7 355 parallel + 0.3 355 senkrecht  
%mean_raw_signal(1,:) = 0.7*mean_raw_signal(1,:) + 0.3*mean_raw_signal(2,:); 
%
for j=1:nchan
bg(j) = mean(mean_raw_bg(j,10:230));  % pt is 0:248 , why taking 10:230 ?
bg_corr(j,:) = mean_signal(j,:) - bg(j);
range_corr(j,:) = range.*range;
%
% Smoothing
bg_corr_sm(j,:) = smooth(bg_corr(j,:),11,'sgolay',3);
end       
%***********************************************
       pr2 = bg_corr.*range_corr;  
%***********************************************
% range corrected signal smoothed over 330 m! 
%***********************************************
    pr2_sm = bg_corr_sm.*range_corr;  
%************************************************ 
   
% Plots
%--------
figure(2)
 set(gcf,'position',[50,100,1000,800]); % units in pixels! *** 19 " ***
%set(gcf,'position',[20,200,500,650]); % units in pixels! *** Laptop ***
% 
subplot(1,2,1) 
subplot('Position',[0.08 0.1 0.4 0.8]);  
  plot(bg_corr_sm(1,:),alt(:),'b','LineWidth',2)  
  hold on
  plot(bg_corr_sm(2,:),alt(:),'b--','LineWidth',2)
  plot(bg_corr_sm(3,:),alt(:),'c','LineWidth',2)
  plot(bg_corr_sm(4,:),alt(:),'k--','LineWidth',2)
  plot(bg_corr_sm(5,:),alt(:),'g','LineWidth',2)
  plot(bg_corr_sm(6,:),alt(:),'g--','LineWidth',2)
  plot(bg_corr_sm(7,:),alt(:),'r','LineWidth',2)
  plot(bg_corr_sm(8,:),alt(:),'m--','LineWidth',2)
  plot(bg_corr_sm(9,:),alt(:),'g-.','LineWidth',2)
  plot(bg_corr_sm(10,:),alt(:),'r-.','LineWidth',2)
  grid on
  title(['Polly^{XT} TROPOS bg-corr signals ' datum ',' ...
       timex(i_start,1:5) '-' timex(i_stop,1:5)],'fontsize',14) 
  xlabel('a. u.','fontsize',14)  
  ylabel('height a.s.l. / m','fontsize',14)
  axis([-0.1 2 0 10e3]); 
  % set(gca,'XTick',[0 0.1 0.2 0.3 0.4 0.5]);
  % set(gca,'XTickLabel',{'0';'0.1';'0.2';'0.3';'0.4';'0.5'},'fontsize',[14])
legend('355', '355 s', '387','407','532','532s','607','1064','532NF','607NF');
%   
subplot(1,2,2)
subplot('Position',[0.52 0.1 0.4 0.8]); 
%
  plot(pr2_sm(1,:),alt(:),'b','LineWidth',2); 
  hold on
  plot(pr2_sm(2,:),alt(:),'b--','LineWidth',2); 
  plot(pr2_sm(3,:),alt(:),'c','LineWidth',2); 
  plot(pr2_sm(4,:),alt(:),'k--','LineWidth',2); 
  plot(pr2_sm(5,:),alt(:),'g','LineWidth',2); 
  plot(pr2_sm(6,:),alt(:),'g--','LineWidth',2); 
  plot(pr2_sm(7,:),alt(:),'r','LineWidth',2); 
  plot(pr2_sm(8,:),alt(:),'m','LineWidth',2); 
  plot(pr2_sm(9,:),alt(:),'g-.','LineWidth',2); 
  plot(pr2_sm(10,:),alt(:),'r-.','LineWidth',2); 
 grid on 
% 
  title('Polly^{XT} TROPOS range corrected signals ' ,'fontsize',14)    
 % datum ', ' timex(i_start,1:5) '-' timex(i_stop,1:5)] 
  xlabel('a.u.','fontsize',14)  
  axis([-1e4 2e5 0 10e3]); 
  % set(gca,'XTick',[0 0.002 0.004 0.006 0.008 0.01],'fontsize',14)
  % set(gca,'XTickLabel',{'0';'2';'4';'6';'8';'10'},'fontsize',14)
% set(gca,'XTickLabel',{'0';'0.002';'0.004';'0.006';'0.008';'0.01'},'fontsi
% ze',[14]
legend('355', '355 s', '387','407','532','532s','607','1064','532NF','607NF');
%   
figure(3)
set(gcf,'position',[50,100,500,800]); % units in pixels! *** 19 " ***
%subplot(1,2,1) 
%subplot('Position',[0.08 0.1 0.4 0.8]); 
  plot(sum_raw_signal(1,1:256),(1:256),'b','LineWidth',2)  
  hold on
  plot(sum_raw_signal(2,1:256),(1:256),'b--','LineWidth',2)
  plot(sum_raw_signal(3,1:256),(1:256),'c','LineWidth',2)
  plot(sum_raw_signal(4,1:256),(1:256),'k--','LineWidth',2)
  plot(sum_raw_signal(5,1:256),(1:256),'g','LineWidth',2)
  plot(sum_raw_signal(6,1:256),(1:256),'g--','LineWidth',2)
  plot(sum_raw_signal(7,1:256),(1:256),'r','LineWidth',2)
  plot(sum_raw_signal(8,1:256),(1:256),'m','LineWidth',2)
  plot(sum_raw_signal(9,1:256),(1:256),'g-.','LineWidth',2)
  plot(sum_raw_signal(10,1:256),(1:256),'r-.','LineWidth',2)
  grid on
  title(['Polly^{XT} Pretrigger ' datum ',' timex(i_start,1:5) ...
      '-' timex(i_stop,1:5)],'fontsize',14) 
  xlabel('','fontsize',14)  
  ylabel('rangebins','fontsize',14)
  axis([-0.1 1.2e4 0 256]); 
  % set(gca,'XTick',[0 0.1 0.2 0.3 0.4 0.5]);
  % set(gca,'XTickLabel',{'0';'0.1';'0.2';'0.3';'0.4';'0.5'},'fontsize',[14])
  legend('355', '355s', '387','407','532','532s','607','1064','532NF','607NF');

 
 
 