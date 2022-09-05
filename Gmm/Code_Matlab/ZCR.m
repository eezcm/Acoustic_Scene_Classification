function [zcr]=ZCR(filename)
[xx,Fs]=audioread(filename);             % 读入数据文件
x=xx-mean(xx);                    % 消除直流分量
wlen=200; inc=80;                 % 设置帧长、帧移
win=hanning(wlen);                % 窗函数
N=length(x);                      % 求数据长度
X=enframe(x,win,inc)';            % 分帧
fn=size(X,2);                     % 获取帧数
zcr=zeros(1,fn);                 % 初始化
for i=1:fn
    z=X(:,i);                     % 取得一帧数据
    for j=1: (wlen- 1);          % 在一帧内寻找过零点
         if z(j)* z(j+1)< 0       % 判断是否为过零点
             zcr(i)=zcr(i)+1;   % 是过零点，记录1次
         end
    end
end
time=(0:N-1)/Fs;                  % 计算时间坐标
frameTime=frame2time(fn,wlen,inc,Fs);  % 求出每帧对应的时间
% 作图
% subplot 211; plot(time,x,'k'); grid;
% title('语音波形');
% ylabel('幅值'); xlabel(['时间/s' 10 '(a)']);
% subplot 212; plot(frameTime,zcr,'k'); grid;
% title('短时平均过零率');
% ylabel('幅值'); xlabel(['时间/s' 10 '(b)']);