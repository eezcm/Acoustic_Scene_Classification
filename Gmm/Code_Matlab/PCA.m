 [coeff,score,latent,tsquared]=pca(mfccs);
rate = cumsum(latent)./sum(latent);%用以确定95%的准确率的列数
tranMatrix=coeff(:,1:22)%经调试发现应该选取为22列
mfcc_result=bsxfun(@minus,mfccs,mean(mfccs,1))*tranMatrix%pca降维结果
save('003_truck_horn.mat','mfcc_result')