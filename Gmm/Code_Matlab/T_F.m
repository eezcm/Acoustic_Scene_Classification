[x,Fs]=audioread('test.wav');
x = x(:,1);
x = x';
N = length(x);%��ȡ��������
t = (0:N-1)/Fs;%��ʾʵ��ʱ��
y = fft(x);%���źŽ��и���Ҷ�任
f = Fs/N*(0:round(N/2)-1);%��ʾʵ��Ƶ���һ�룬Ƶ��ӳ�䣬ת��ΪHZ
subplot(211);
plot(t,x,'g');%����ʱ����
xlabel('Time/s');ylabel('Amplitude');
title('�źŵĲ���');
grid;
subplot(212);
plot(f,abs(y(1:round(N/2))));
xlabel('Frequency/Hz');ylabel('Amplitude');
title('�źŵ�Ƶ��');
grid;
