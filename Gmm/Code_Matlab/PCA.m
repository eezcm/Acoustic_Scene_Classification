 [coeff,score,latent,tsquared]=pca(mfccs);
rate = cumsum(latent)./sum(latent);%����ȷ��95%��׼ȷ�ʵ�����
tranMatrix=coeff(:,1:22)%�����Է���Ӧ��ѡȡΪ22��
mfcc_result=bsxfun(@minus,mfccs,mean(mfccs,1))*tranMatrix%pca��ά���
save('003_truck_horn.mat','mfcc_result')