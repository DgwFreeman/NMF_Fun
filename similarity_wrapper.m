DataDir = 'C:\Users\Freeman\Documents\GitHub\NMF_Fun\Results\NMF_Similarity\';
Filelist = dir(fullfile(DataDir,'*.mat'));
load(fullfile(DataDir,Filelist(1).name));

noise = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3];
nNoise = length(noise);
nXVAL = 10; 
cc = lines(nNoise);
for iN = 1:nNoise
    xvNMF = rrNMF_xv{iN,1};
    SimilarityScore = similarity_analysis(xvNMF);

    for iXV = 1:nXVAL
        ExtrFeat = xvNMF{iXV,1};
        nFeat(iXV) = length(ExtrFeat);
    end
    maxFeat = max(nFeat);

    mSS = [];
    stdSS = [];
    for iK = 1:maxFeat
        pos = SimilarityScore(iK,:) ~= 1;
        mSS(iK) = mean(SimilarityScore(iK,pos));
        stdSS(iK) = std(SimilarityScore(iK,pos));
%         plot(iK,SimilarityScore(iK,pos),'.k','MarkerSize',10),hold on
    end
    errorbar(1:maxFeat,mSS,stdSS,'-o','LineWidth',2,'Color',cc(iN,:)),hold on

    legendStr{iN} = sprintf('%u%% Noise',int16(noise(iN)*100));

end

xlabel('Number of Components');
ylabel('Similarity Score');
xlim([0,8])
legend(legendStr);
