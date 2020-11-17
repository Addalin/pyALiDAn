function gammln(xx)
% aus NR
% returens the value ln(gammma(xx) for xx > 0
cof = [76.18009173,-86.50532033,24.01409822,...
        -1.231739516, 0.120858003e-2, -0.536382e-5]; 
stp =  2.50662827465;

x = xx; 
y = x; 
tmp = x+5.5d0; 
tmp = (x+0.5d0)*log(tmp)-tmp;

ser = 1.000000000190015d0; 
    for j = 1:6
        y = y +1.0d0; 
        ser = ser+cof(j)/y;
    end 
   gammln = tmp+log(stp*ser/x);    
   
 % in FORTRAN  
 %   FUNCTION GAMMLN(XX)
 %     REAL*8 COF(6),STP,HALF,ONE,FPF,X,TMP,SER
 %     DATA COF,STP/76.18009173D0,-86.50532033D0,24.01409822D0,
 %    +    -1.231739516D0,.120858003D-2,-.536382D-5,2.50662827465D0/
 %     DATA HALF,ONE,FPF/0.5D0,1.0D0,5.5D0/
 %
 %     X=XX-ONE
 %     TMP=X+FPF
 %     TMP=(X+HALF)*LOG(TMP)-TMP
 %     SER=ONE
 %     DO 11 J=1,6
 %       X=X+ONE
 %       SER=SER+COF(J)/X
%11    CONTINUE
 %     GAMMLN=TMP+LOG(STP*SER)
 %
 %     RETURN
 %     END