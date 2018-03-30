% DataDir = 'C:\Users\Freeman\Documents\GitHub\NMF_Fun\Results\NMF_Similarity\K_MeansInit\';
% Filelist = dir(fullfile(DataDir,'*.mat'));
% load(fullfile(DataDir,Filelist(2).name));
load('./XvalSimilarity_Rand.mat');

noise = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3];
nNoise = length(noise);
nXVAL = 10; 
cc = lines(nNoise);
for iN = 4:nNoise
    xvNMF = rrNMF_xv{iN,1};
    [SimilarityScore,sqerr_te] = similarity_analysis(xvNMF);
   
    for iXV = 1:nXVAL
        ExtrFeat = xvNMF{iXV,1};
        nFeat(iXV) = length(ExtrFeat);
        DC(:,iXV) = xvNMF{iXV,4};
    end
    maxFeat = max(nFeat);

    mSS = [];
    stdSS = [];
    
    for iK = 1:maxFeat
        pos = SimilarityScore(iK,:) ~= 1;
        mSS(iK) = mean(SimilarityScore(iK,pos));
        stdSS(iK) = std(SimilarityScore(iK,pos));
        
        mSQE(iK) = mean(sqerr_te(iK,:));
        stdSQE(iK) = std(sqerr_te(iK,:));
        
        mmDC(iK,1) = mean(DC(iK,:));
        ssDC(iK,1) = std(DC(iK,:));
        plot(iK,SimilarityScore(iK,pos)*100,'.r','MarkerSize',10,'Color',cc(iN,:)),hold on
    end
    a1 = plot(1:maxFeat,mSS*100,'-o','LineWidth',2,'Color',cc(iN,:)),hold on
    legendStr{iN} = sprintf('%u%% Noise',int16(noise(iN)*100));
    a2 = plot(1:maxFeat,mSQE*100,'-o','LineWidth',2,'Color',cc(7,:)),hold on
    a3 = plot(1:maxFeat,mmDC,'-x','LineWidth',2,'Color',cc(5,:)),hold on
    xlabel('Number of Components');
    ax = gca;
    ax.XGrid = 'on';
    ylabel('Percentage');
    legend([a1,a2,a3],'Similarity Score','Reconstruction Error','Decoding Performance')
    
end



xlabel('Number of Components');
ylabel('Percentage');
xlim([0,8])
legend('Similarity Score','Reconstruction Error','Decoding Performance')


