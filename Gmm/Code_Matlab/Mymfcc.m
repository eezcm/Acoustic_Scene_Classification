function [mfcc_final]=Mymfcc(filename)

frameSize=512;
inc=160;
[x,fs]=audioread(filename);%��ȡwav�ļ�
N=length(x);
%Ԥ����y=x(i)-0.97*x(i-1)
for i=2:N
    y(i)=x(i)-0.97*x(i-1);
end
y=y';%��yȡת��
S=enframe(y,frameSize,inc);%��֡,��x���з�֡��
[a b]=size(S);
%����һ�º�����a=0.46,�õ�������W=(1-a)-a*cos(2*pi*n/N)
n=1:b;
W=0.54-0.46*cos((2*pi.*n)/b);
%��������������C
C=zeros(a,b);
ham=hamming(b);
for i=1:a
    C(i,:)=ham;
end
%��������C��S��˵�SC
SC=S.*C;
F=0;N=4096;
for i=1:a
    %��SC��N=4096��FFT�任
    D(i,:)=fft(SC(i,:),N);
    %����ѭ��ʵ����ȡ�������ܶ�E
    for j=1:N
        t=abs(D(i,j));
        E(i,j)=(t^2)/N;
    end
    %��ȡÿһ֡�������ܺ�F(i)
    F(i)=sum(D(i,:));
end

%��Ƶ��ת��Ϊ÷��Ƶ��
%÷��Ƶ��ת������ͼ��
N1=length(x)
for i=1:N1
    mel(i)=2595*log10(1+i/700);
end

fl=0;fh=fs/2;%����Ƶ�ʷ�Χ����Ƶ�͸�Ƶ
bl=2595*log10(1+fl/700);%�õ�÷���̶ȵ���Сֵ
bh=2595*log10(1+fh/700);%�õ�÷���̶ȵ����ֵ
%÷�����귶Χ
p=26;%�˲�������
B=bh-bl;%÷���̶ȳ���
mm=linspace(0,B,p+2);%�滮28����ͬ��÷���̶�
fm=700*(10.^(mm/2595)-1);%��MelƵ��ת��ΪƵ��
W2=N/2+1;%fs/2�ڶ�Ӧ��FFT����,2049��Ƶ�ʷ���

k=((N+1)*fm)/fs%����28����ͬ��kֵ
hm=zeros(26,N);%����hm����
df=fs/N;
freq=(0:N-1)*df;%����Ƶ��ֵ

%����÷���˲���
for i=2:27
    %ȡ��������ȡ����28��k�еĵ�2-27��������0��28
    n0=floor(k(i-1));
    n1=floor(k(i));
    n2=floor(k(i+1));
    %k(i)�ֱ�������ÿ��÷��ֵ���µķ�Χ�ڵ�ӳ�䣬��ȡֵ��ΧΪ��0-N/2
    %������ȡ�����˲�����Ƶ����Ӧ��
   for j=1:N
       if n0<=j & j<=n1
           hm(i-1,j)=2*(j-n0)/((n2-n0)*(n1-n0));
       elseif n1<=j & j<=n2
           hm(i-1,j)=2*(n2-j)/((n2-n0)*(n1-n0));
       end
   end
   %�˴���ȡH1(k)������
end
%��ͼ,��ÿ����ɫ��ʾ��һ��
c=colormap(lines(26));%����26����ͬ��ɫ������
figure(1)
set(gcf,'position',[180,160,1020,550]);%���û�ͼ�Ĵ�С
for i=1:26
    plot(freq,hm(i,:),'--','color',c(i,:),'linewidth',2.5);%��ʼѭ������ÿ��÷���˲���
    hold on
end
 grid on;%��ʾ����
 axis([0 1500 0 0.1]);%������ʾ��Χ
 
 %�õ��������������ĺ�
 H=E*hm';
 %��H����Ȼ��������
 %��Ϊ�˶��������������źű���Ĵ�С���ݴη���ϵ������Ҫ�������
 for i=1:a
     for j=1:26
         H(i,j)=log(H(i,j));
     end
 end
 %����ɢ���ұ任
 for i=1:a
     for j=1:26
         %����ȡÿһ֡�������ܺ�
         sum1=0;
         %����ɢ���ұ任
         for p=1:26
             sum1=sum1+H(i,p)*cos((pi*j)*(2*p-1)/(2*26));
         end
         mfcc(i,j)=((2/26)^0.5)*sum1;  
         %�����ɢ���ұ任
     end    
 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%   ����Ϊ��ȡmfcc��������������  %%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %������������
 %��Ϊ�󲿷ֵ��ź�����һ�㼯���ڱ任��ĵ�Ƶ�������Զ�ÿһֻ֡ȡǰ12������
 J=mfcc(:,(1:12));
 %Ĭ��������ϵ��Ϊ22
 for i=1:12
     K(i)=1+(22/2)*sin(pi*i/22);
 end
 %�õ���ά����feat,����mfcc�ĵ�һ�����ݣ�Ĭ��Ϊ����
 for i=1:a
     for j=1:12
         L(i,j)=J(i,j)*K(j);
     end
 end
 feat=L;
 %��������ȡ�ڶ��飨һ�ײ��ϵ���� ����Ҳ��mfcc�����ĵڶ������
 dtfeat=0;
 dtfeat=zeros(size(L));%Ĭ�ϳ�ʼ��
 for i=3:a-2
     dtfeat(i,:)=-2*feat(i-2,:)-feat(i-1,:)+feat(i+1,:)+2*feat(i+2,:); 
 end
%��ȡ���ײ��ϵ��,mfcc�����ĵ��������
%���ײ��ϵ�����Ƕ�ǰ�������һ�ײ��ϵ��dtfeat�ٴν��в�����
 dttfeat=0;
 dttfeat=zeros(size(dtfeat));%Ĭ�ϳ�ʼ��
 for i=3:a-2
     dttfeat(i,:)=-2*dtfeat(i-2,:)-dtfeat(i-1,:)+dtfeat(i+1,:)+2*dtfeat(i+2,:); 
 end
 dttfeat=dttfeat/10;
 %�����10�Ǹ�������ȷ���ģ�Ĭ��Ϊ2
 %���õ���mfcc����������feat��dtfeat��dttfeatƴ�ӵ�һ��
 %�õ�����mfccϵ��
 mfcc_final=0;
 mfcc_final=[feat,dtfeat,dttfeat];%ƴ�����

 