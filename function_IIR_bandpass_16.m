function y=function_IIR_bandpass_16(x,fp,fst,Fs)
%length*16
Rp=1;
As=40;

%ȥֱ������

    for i=1:16
        x(i,:)=x(i,:)-mean(x(i,:));
    end
    omega_p=fp/Fs*2*pi;
    omega_st=fst/Fs*2*pi;

    %���һ��IIR filter
    [N1,Wp1] = ellipord(omega_p/pi,omega_st/pi,Rp,As,'s');
    [B,A] = ellip(N1,Rp,As,Wp1,'low');

    %���źŽ����˲�
    for i=1:16
        y(i,:)=filter(B,A,x(i,:));
    end
    
    