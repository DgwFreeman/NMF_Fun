function [SimilarityScore,sqerr_te] = similarity_analysis(xvNMF)

nXVAL = length(xvNMF);
nFeat = zeros(nXVAL,1);

%Find max # of features extracted out of all the x-validation runs
for iXV = 1:nXVAL
    ExtrFeat = xvNMF{iXV,1};
    nFeat(iXV) = length(ExtrFeat);
end
maxFeat = max(nFeat);

%% Find the reconstruction error for each value of k in each xval run
sqerr_tr = NaN(maxFeat,nXVAL);
sqerr_te = NaN(maxFeat,nXVAL);
SimilarityScore = NaN(maxFeat,nXVAL);
for iXV = 1:nXVAL
    ExtrFeat = xvNMF{iXV,1}; 
    X_train = xvNMF{iXV,2}; 
    X_test = xvNMF{iXV,3}; 
    
    for iK = 1:nFeat(iXV)
        W = ExtrFeat{iK,1};
        H_train = ExtrFeat{iK,2};
        H_test = ExtrFeat{iK,3};
        
        sqerr_tr(iK,iXV) = norm(X_train - W*H_train,'fro')^2/norm(X_train,'fro')^2;
        sqerr_te(iK,iXV) = norm(X_test - W*H_test,'fro')^2/norm(X_test,'fro')^2;
    end
end
%Loop through each value of k to perform a similarity analysis
for iK = 1:maxFeat
     
    %Which run should we compare the rest to?
    [mSQE, iSQE] = min(sqerr_te(iK,:));
    W1 = xvNMF{iSQE,1}{iK,1};
    H1 = xvNMF{iSQE,1}{iK,3};
    
    for iXV = 1:nXVAL
        if iK > nFeat(iXV), continue;end
        ExtrFeat = xvNMF{iXV,1};
        if iXV == iSQE
            SimilarityScore(iK,iXV) = 1;
        else
            W2 = xvNMF{iXV,1}{iK,1};
            H2 = xvNMF{iXV,1}{iK,3};
            SimilarityScore(iK,iXV) = similarity(W1,H1,W2,H2);
        end
        
    end
end

%% Plot 
% figure
% for iK = 1:maxFeat
%     pos = SimilarityScore(iK,:) ~= 1;
%     mSS(iK) = mean(SimilarityScore(iK,pos));
%     plot(iK,SimilarityScore(iK,pos),'.k','MarkerSize',10),hold on
% end
% plot(1:maxFeat,mSS,'-b','LineWidth',2),hold on
% xlabel('Number of Components');
% ylabel('Similarity Score');
% xlim([0,8])
% title('20% Additive Noise')

end

