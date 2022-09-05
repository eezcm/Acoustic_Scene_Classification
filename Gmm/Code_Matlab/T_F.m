[x,Fs]=audioread('test.wav');
x = x(:,1);
x = x';
N = length(x);%求取抽样点数
t = (0:N-1)/Fs;%显示实际时间
y = fft(x);%对信号进行傅里叶变换
f = Fs/N*(0:round(N/2)-1);%显示实际频点的一半，频域映射，转化为HZ
subplot(211);
plot(t,x,'g');%绘制时域波形
xlabel('Time/s');ylabel('Amplitude');
title('信号的波形');
grid;
subplot(212);
plot(f,abs(y(1:round(N/2))));
xlabel('Frequency/Hz');ylabel('Amplitude');
title('信号的频谱');
grid;
