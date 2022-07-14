function [mfcc_final]=Mymfcc(filename)

frameSize=512;
inc=160;
[x,fs]=audioread(filename);%读取wav文件
N=length(x);
%预加重y=x(i)-0.97*x(i-1)
for i=2:N
    y(i)=x(i)-0.97*x(i-1);
end
y=y';%对y取转置
S=enframe(y,frameSize,inc);%分帧,对x进行分帧，
[a b]=size(S);
%尝试一下汉明窗a=0.46,得到汉明窗W=(1-a)-a*cos(2*pi*n/N)
n=1:b;
W=0.54-0.46*cos((2*pi.*n)/b);
%创建汉明窗矩阵C
C=zeros(a,b);
ham=hamming(b);
for i=1:a
    C(i,:)=ham;
end
%将汉明窗C和S相乘得SC
SC=S.*C;
F=0;N=4096;
for i=1:a
    %对SC作N=4096的FFT变换
    D(i,:)=fft(SC(i,:),N);
    %以下循环实现求取能量谱密度E
    for j=1:N
        t=abs(D(i,j));
        E(i,j)=(t^2)/N;
    end
    %获取每一帧的能量总和F(i)
    F(i)=sum(D(i,:));
end

%将频率转换为梅尔频率
%梅尔频率转化函数图像
N1=length(x)
for i=1:N1
    mel(i)=2595*log10(1+i/700);
end

fl=0;fh=fs/2;%定义频率范围，低频和高频
bl=2595*log10(1+fl/700);%得到梅尔刻度的最小值
bh=2595*log10(1+fh/700);%得到梅尔刻度的最大值
%梅尔坐标范围
p=26;%滤波器个数
B=bh-bl;%梅尔刻度长度
mm=linspace(0,B,p+2);%规划28个不同的梅尔刻度
fm=700*(10.^(mm/2595)-1);%将Mel频率转换为频率
W2=N/2+1;%fs/2内对应的FFT点数,2049个频率分量

k=((N+1)*fm)/fs%计算28个不同的k值
hm=zeros(26,N);%创建hm矩阵
df=fs/N;
freq=(0:N-1)*df;%采样频率值

%绘制梅尔滤波器
for i=2:27
    %取整，这里取得是28个k中的第2-27个，舍弃0和28
    n0=floor(k(i-1));
    n1=floor(k(i));
    n2=floor(k(i+1));
    %k(i)分别代表的是每个梅尔值在新的范围内的映射，其取值范围为：0-N/2
    %以下求取三角滤波器的频率响应。
   for j=1:N
       if n0<=j & j<=n1
           hm(i-1,j)=2*(j-n0)/((n2-n0)*(n1-n0));
       elseif n1<=j & j<=n2
           hm(i-1,j)=2*(n2-j)/((n2-n0)*(n1-n0));
       end
   end
   %此处求取H1(k)结束。
end
%绘图,且每条颜色显示不一样
c=colormap(lines(26));%定义26条不同颜色的线条
figure(1)
set(gcf,'position',[180,160,1020,550]);%设置画图的大小
for i=1:26
    plot(freq,hm(i,:),'--','color',c(i,:),'linewidth',2.5);%开始循环绘制每个梅尔滤波器
    hold on
end
 grid on;%显示方格
 axis([0 1500 0 0.1]);%设置显示范围
 
 %得到能量特征参数的和
 H=E*hm';
 %对H作自然对数运算
 %因为人耳听到的声音与信号本身的大小是幂次方关系，所以要求个对数
 for i=1:a
     for j=1:26
         H(i,j)=log(H(i,j));
     end
 end
 %作离散余弦变换
 for i=1:a
     for j=1:26
         %先求取每一帧的能量总和
         sum1=0;
         %作离散余弦变换
         for p=1:26
             sum1=sum1+H(i,p)*cos((pi*j)*(2*p-1)/(2*26));
         end
         mfcc(i,j)=((2/26)^0.5)*sum1;  
         %完成离散余弦变换
     end    
 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%   以下为求取mfcc的三个参数过程  %%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %作升倒谱运算
 %因为大部分的信号数据一般集中在变换后的低频区，所以对每一帧只取前12个数据
 J=mfcc(:,(1:12));
 %默认升到普系数为22
 for i=1:12
     K(i)=1+(22/2)*sin(pi*i/22);
 end
 %得到二维数组feat,这是mfcc的第一组数据，默认为三组
 for i=1:a
     for j=1:12
         L(i,j)=J(i,j)*K(j);
     end
 end
 feat=L;
 %接下来求取第二组（一阶差分系数） ，这也是mfcc参数的第二组参数
 dtfeat=0;
 dtfeat=zeros(size(L));%默认初始化
 for i=3:a-2
     dtfeat(i,:)=-2*feat(i-2,:)-feat(i-1,:)+feat(i+1,:)+2*feat(i+2,:); 
 end
%求取二阶差分系数,mfcc参数的第三组参数
%二阶差分系数就是对前面产生的一阶差分系数dtfeat再次进行操作。
 dttfeat=0;
 dttfeat=zeros(size(dtfeat));%默认初始化
 for i=3:a-2
     dttfeat(i,:)=-2*dtfeat(i-2,:)-dtfeat(i-1,:)+dtfeat(i+1,:)+2*dtfeat(i+2,:); 
 end
 dttfeat=dttfeat/10;
 %这里的10是根据数据确定的，默认为2
 %将得到的mfcc的三个参数feat、dtfeat、dttfeat拼接到一起
 %得到最后的mfcc系数
 mfcc_final=0;
 mfcc_final=[feat,dtfeat,dttfeat];%拼接完成

 