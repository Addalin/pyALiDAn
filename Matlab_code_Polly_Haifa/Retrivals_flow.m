
%% Load data (.nc files)  from F:\data_haifa and doing background correction (remove the pre-triggered signal) 
A_read_pollyXT_cnt80_Haifa 

%% Calculating the molecular bacskater from files at : F:\data_haifa\Radiosonden\
B_read_sonde_pollyXT_cnt0   
% Radiosonden Files can be download from http://weather.uwyo.edu/upperair/sounding.html.
% Remember to remove the lower part of the txt file, and rename the file name : 40179_yyyymmdd_hh.dat  
% Lidar Ratio - 55 is valid for urban aerosol, and for saharn dust - this is an assupsion. 

%% Find the refence range. 
C_rayleigh_fit_PollyXT_cnt80  
% The refernce range should be where the SNR ratio - at least 3
% molecular fit for substruction.  
%Find and playaround with the regference range (l.16-23) , and the threashholds of the differences  (l. 51-135)

%% Calculate aerosole backscatter coef.
D_Klett_pollyXT_cnt80 
% Chek wavelength behavoiure - R->G->B  , and above the aerosoles the profile should be zero

%% Calculate aerosole extiction coef.
E_Raman_pollyXT_cnt80 

%%
F_Raman_beta_pollyXT_Haifa
%F_Raman_beta_pollyXT_cnt80 %Brigit will send in few days