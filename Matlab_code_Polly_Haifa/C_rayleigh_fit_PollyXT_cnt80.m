% rayleigh_fit_PollyXT.m 
%
%  BHeese 08/07
%   
% vorher folgende M-Files laufen lassen:  
%
%       read_pollyXT.m 
%       
% -------------------------------------------------
% Rayleigh Signal auf das Lidar Signal scalieren
% -------------------------------------------------
% Find and playaround with the regference range (l.16-23) , and the
% threashholds of the differences  (l. 51-135) - ADI
clear RefBin
%
  altbin=sin((90-zenithangle)*(pi/180))*(c*measurement_height_resolution*1e-12)/2;
%  Elastische Kanäle 
  xl_scal_1 = round(5.8/altbin);
  xu_scal_1 = round(6.5/altbin);
%  Depol Kanäle
  xl_scal_2 = round(5.8/altbin);
  xu_scal_2 = round(6.5/altbin);
%  Raman Kanäle7
  xl_scal_3 = round(5.8/altbin);
  xu_scal_3 = round(6.5/altbin);
% 
% --------------------
%   Signale mitteln 
% --------------------   
int_raw_signal  = sum_raw_signal/nmeas(2); 
%
for i=1:nchan-5 
     meanRaySig(i) = mean(pr2_ray_sig(i,xl_scal_1:xu_scal_1)); 
%    
     mean_pr2(i) = mean(pr2(i,xl_scal_1:xu_scal_1)); 
%    
     SigFak(i) = mean_pr2(i)/meanRaySig(i);  
%
     RaySig(i,:) = SigFak(i).*pr2_ray_sig(i,:); 
%------------------
%  logarithmieren
% -----------------
        Ray_Fit(i,:) = log(RaySig(i,:));  
            
        log_pr2 = real(log(pr2));   
%        log_pr2_sm = log(max(1e-10,pr2_sm));     
end   
%   Referenzbin finden, das auf dem Signal liegt
% -------------------------------------------------
%% *****************
%  Kanal 1  355
% *****************
   abst_1=1e-3; 
%   diff_1 = zeros(1,xl_scal_1:xu_scal_1);
      for j=xl_scal_1:xu_scal_1
          diff_1(j) = (real(log_pr2(1,j)) - Ray_Fit(1,j)).^2; 
               if diff_1(j) < abst_1
                  abst_1 = diff_1(j);
                  RefBin(1)=j;  
               end
      end
% *****************
%    Kanal 2    355s 
% *****************
     abst_2=1e-3; 
%     diff_2= zeros(1,xl_scal_2:xu_scal_2);
       for j=xl_scal_1:xu_scal_1 
           diff_2(j) = (real(log_pr2(2,j))- Ray_Fit(2,j)).^2;  
                if diff_2(j) < abst_2
                  abst_2=diff_2(j);
                  RefBin(2)=j; 
               end
       end
% *****************
%    Kanal 3    387
% *****************
     abst_3=1e-3; 
%     diff_3=zeros(1,xl_scal_3:xu_scal_3);
       for j=xl_scal_1:xu_scal_1 
           diff_3(j) = (real(log_pr2(3,j))- Ray_Fit(3,j)).^2;  
                if diff_3(j) < abst_3
                  abst_3=diff_3(j); 
                  RefBin(3)=j; 
               end
       end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       
%if hour(i_start) ~= 0 & hour(i_start) >= 6 & hour(i_start) <= 18 
%       RefBin(3) = 100;  % bei 387 am Tage 
%end 
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% *****************
%  Kanal 5 532
% *****************
   abst_5=1e-2; 
%diff_4=zeros(1,xl_scal_1:xu_scal_1);
      for j=xl_scal_1:xu_scal_1
          diff_5(j) = (log_pr2(5,j) - Ray_Fit(5,j)).^2;  
               if diff_5(j) < abst_5
                  abst_5 = diff_5(j);
                  RefBin(5)=j;  
               end
      end
% *****************
%    Kanal 7 607
% *****************
     abst_7=1e-3; 
%     diff_7 = zeros(1,xl_scal_3:xu_scal_3);
       for j=xl_scal_1:xu_scal_1 
           diff_7(j) = (real(log_pr2(7,j))- Ray_Fit(7,j)).^2; 
                if diff_7(j) < abst_7
                  abst_7=diff_7(j);
                  RefBin(7)=j; 
                end
              
       end      
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
%if hour(i_start) ~= 0 & hour(i_start) >= 6 & hour(i_start) <= 18 
%         RefBin(5) = 100;  % bei 607 am Tage       
%end
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% *****************
%  Kanal 8 1064
% *****************
   abst_8=1e-3; 
%diff_6= zeros(1,xl_scal_1:xu_scal_1);  % this is the fit of rayley
%calculation and mulecular profile. 
      for j=xl_scal_1:xu_scal_1
          diff_8(j) = (real(log_pr2(8,j)) - Ray_Fit(8,j)).^2;  % are these the differences from p.29  in L.4 - Adi
               if diff_8(j) < abst_8
                  abst_8 = diff_8(j);
                  RefBin(8)=j; 
               end
      end
%
% --------      
%  Plots
% --------
rb = 1500;  % Plothöhe

figure(90)
  set(gcf,'position',[50,100,1200,800]); % units in pixels! *** 19 " ***
% set(gcf,'position',[20,200,1200,650]); % units in pixels! *** Laptop ***
%  title(['Polly backscatter 532 nm and Raman signal 607 nm ' date(1) ' from ' ...
%       ' UTC'],'fontsize',[13]) 
%
 subplot(1,6,1) %355
  subplot('Position',[0.04 0.08 0.13 0.85]); 
%  xlabel('a.u.','fontsize',12)  
  ylabel('height / m','fontsize',12)
  axis([-1e4 max(pr2(1,1:rb))+0.1.*max(pr2(1,1:rb)) 0 alt(1,rb)]); 
  box on 
  hold on
  plot(pr2(1,1:rb), alt(1,1:rb)); 
  plot(RaySig(1,1:rb), alt(1,1:rb),'g','LineWidth',2); 
  plot(pr2_sm(1,RefBin(1)), alt(1,RefBin(1)),'r*');
%  legend(h,'pr2-Analog','pr2-PC','Rayleigh Fit-Analog','Rayleigh Fit-PC'); 
%  legend(h,'pr2-Analog',['Rayleigh Fit: ' radiofile(46:53)]); 
  grid on
  legend('355 nm', 'Rayleigh Fit', 'Referenz Bin'); 
%
%   
subplot(1,6,2) %355s
 subplot('Position',[0.2 0.08 0.13 0.85]); 
%   xlabel('a.u.','fontsize',12)  
%  ylabel('height / m','fontsize',12)
  axis([-0.5e4 max(pr2_sm(2,1:rb))+0.1*max(pr2_sm(2,1:rb)) 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(pr2(2,1:rb), alt(1,1:rb)); 
  plot(RaySig(2,1:rb), alt(1,1:rb),'g','LineWidth',2); 
  plot(pr2(2,RefBin(2)), alt(RefBin(2)),'r*');
  legend('355 nm s', 'Rayleigh Fit', 'Referenz Bin'); 
  grid on
 
  subplot(1,6,3)    %387
 subplot('Position',[0.36 0.08 0.13 0.85]); 
   title('Pr^2','fontsize',12) 
%   xlabel('a.u.','fontsize',12)  
%  ylabel('height / m','fontsize',12)
  axis([-0.5e4 max(pr2(3,1:rb))+0.1*max(pr2(3,1:rb)) 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(pr2(3,1:rb), alt(1,1:rb)); 
  plot(RaySig(3,1:rb), alt(1,1:rb),'g','LineWidth',2); 
  plot(pr2(3,RefBin(3)), alt(RefBin(3)),'r*');
  legend('387 nm', 'Rayleigh Fit', 'Referenz Bin'); 
  grid on

  subplot(1,6,4) %532
  subplot('Position',[0.52 0.08 0.13 0.85]); 
  title([timex(i_start,1:5) ' - ' timex(i_stop,1:5) ' UTC '],'fontsize',12) 
%   xlabel('a.u.','fontsize',12)  
%  ylabel('height / m','fontsize',12)
  axis([-0.5e4 max(pr2(5,1:rb))+0.1*max(pr2(5,1:rb)) 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(pr2(5,1:rb), alt(1,1:rb)); 
  plot(RaySig(5,1:rb), alt(1,1:rb),'g','LineWidth',2); 
  plot(pr2(5,RefBin(5)), alt(RefBin(5)),'r*');
  legend('532 nm', 'Rayleigh Fit', 'Referenz Bin'); 
  grid on

  subplot(1,6,5) % 607
 subplot('Position',[0.67 0.08 0.13 0.85]); 
% title(datum,'fontsize',12) 
%   xlabel('a.u.','fontsize',12)  
%  ylabel('height / m','fontsize',12)
  axis([-0.5e4 max(pr2(7,1:rb))+0.1*max(pr2(7,1:rb)) 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(pr2(7,1:rb), alt(1,1:rb)); 
  plot(RaySig(7,1:rb), alt(1,1:rb),'g','LineWidth',2); 
  plot(pr2(7,RefBin(7)), alt(RefBin(7)),'r*');
  legend('607 nm', 'Rayleigh Fit', 'Referenz Bin'); 
  grid on

  subplot(1,6,6) %1064
 subplot('Position',[0.83 0.08 0.13 0.85]); 
% title(datum,'fontsize',12) 
%   xlabel('a.u.','fontsize',[12])  
%  ylabel('height / m','fontsize',[12])
  axis([-0.5e4 max(pr2(8,1:rb))+0.1*max(pr2(8,1:rb)) 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(pr2(8,1:rb), alt(1,1:rb)); 
  plot(RaySig(8,1:rb), alt(1,1:rb),'g','LineWidth',2); 
  plot(pr2(8,RefBin(8)), alt(RefBin(8)),'r*');
  legend('1064 nm', 'Rayleigh Fit', 'Referenz Bin'); 
  grid on

 % -------------
 % log signal
 % -------------
figure(100)
 set(gcf,'position',[50,100,1200,800]); % units in pixels! *** 19 " ***
% set(gcf,'position',[20,200,1200,650]); % units in pixels! *** Laptop ***
%  title(['Polly backscatter 532 nm and Raman signal 607 nm ' date(1) ' from ' ...
%       ' UTC'],'fontsize',[13]) 
%
subplot(1,6,1)
subplot('Position',[0.04 0.08 0.13 0.85]); 
 %title('ln Pr^2' ,'fontsize',12) 
 % xlabel('a.u.','fontsize',12)  
 % ylabel('height / m','fontsize',12)
  axis([5 12 0 alt(1,rb)]); 
  box on  
  hold on
  plot(log_pr2(1,1:rb),alt(1,1:rb),'b');    
  plot(Ray_Fit(1,1:rb),alt(1,1:rb),'g','LineWidth',2);   
  plot(log_pr2(1,RefBin(1)), alt(RefBin(1)),'r*');
  grid on
  legend('355 nm', 'Rayleigh Fit', 'Referenz Bin'); 
%   
 subplot(1,6,2)
  subplot('Position',[0.2 0.08 0.13 0.85]); 
  title('ln Pr^2' ,'fontsize',12) 
  %xlabel('','fontsize',12)  
  axis([5 12 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(log_pr2(2,1:rb),alt(1,1:rb),'b');  
  plot(Ray_Fit(2,1:rb),alt(1,1:rb),'g','LineWidth',2);   
  plot(log_pr2(2,RefBin(2)), alt(1,RefBin(2)),'r*');
  grid on
  legend('355 nm s', 'Rayleigh Fit', 'Referenz Bin'); 
  
 subplot(1,6,3)
  subplot('Position',[0.36 0.08 0.13 0.85]); 
  title(datum,'fontsize',12) 
  %xlabel('','fontsize',12)  
  axis([5 12 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(log_pr2(3,1:rb),alt(1,1:rb),'b');  
  plot(Ray_Fit(3,1:rb),alt(1,1:rb),'g','LineWidth',2);   
  plot(log_pr2(3,RefBin(3)), alt(1,RefBin(3)),'r*');
  grid on
  legend('387 nm', 'Rayleigh Fit', 'Referenz Bin'); 

  subplot(1,6,4)
  subplot('Position',[0.52 0.08 0.13 0.85]); 
  title([timex(i_start,1:5) ' - ' timex(i_stop,1:5) ' UTC '],'fontsize',12) 
  %xlabel('','fontsize',[12])  
  axis([5 12 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(log_pr2(5,1:rb),alt(1,1:rb),'b');  
  plot(Ray_Fit(5,1:rb),alt(1,1:rb),'g','LineWidth',2);   
  plot(log_pr2(5,RefBin(5)), alt(1,RefBin(5)),'r*');
  grid on
  legend('532 nm', 'Rayleigh Fit', 'Referenz Bin'); 
  
  subplot(1,6,5)
  subplot('Position',[0.67 0.08 0.13 0.85]); 
 % title('ln Pr^2' ,'fontsize',12) 
  %xlabel('','fontsize',12)  
  axis([5 12 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(log_pr2(7,1:rb),alt(1,1:rb),'b');  
  plot(Ray_Fit(7,1:rb),alt(1,1:rb),'g','LineWidth',2);   
  plot(log_pr2(7,RefBin(7)), alt(1,RefBin(7)),'r*');
  grid on
  legend('607 nm', 'Rayleigh Fit', 'Referenz Bin'); 
  
  subplot(1,6,6)
  subplot('Position',[0.83 0.08 0.13 0.85]); 
% title('ln Pr^2' ,'fontsize',12) 
  %xlabel('','fontsize',12)  
  axis([5 12 0 alt(1,rb)]); 
%  set(gca,'YTickLabel',{''})
  box on 
  hold on
  plot(log_pr2(8,1:rb),alt(1,1:rb),'b');  
  plot(Ray_Fit(8,1:rb),alt(1,1:rb),'g','LineWidth',2);   
  plot(log_pr2(8,RefBin(8)), alt(1,RefBin(8)),'r*');
  grid on
  legend('1064 nm', 'Rayleigh Fit', 'Referenz Bin'); 
  
%  figure(12)
%  plot(log_pr2(1,1:rb)./log_pr2(3,1:rb),alt(1,1:rb),'b');  
%  hold on
%  plot(log_pr2(4,1:rb)./log_pr2(5,1:rb),alt(1,1:rb),'g');  
%  plot(log_pr2(6,1:rb)./log_pr2(5,1:rb),alt(1,1:rb),'r');  
%  axis([0.7 1.3 0 12000]); 
%  legend('355/387', '532/607', '1064/607')
 
  disp('End of program: rayleigh_fit_Polly.m, Vers. 1.0 04/07')
  
