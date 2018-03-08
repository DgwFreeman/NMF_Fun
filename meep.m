clear all
cd 'C:\Users\Freeman\Documents\GitHub\NMF_Fun'
load('.\Results\NMF_Similarity\K_MeansInit\SimulatedData_201803062128.mat',...
    'data','fCoding','noise','nCoding','nNoise','NonCodingPattern','nPatterns',...
    'nStimuli','nTrials','Patterns','nNeurons','nBins');
NoInit = load('.\Results\NMF_Similarity\No_K_Means\NMFResults_201803062258.mat',...
                'kFeat','mDC','stdDC','SpatialModules','ActCoeff');
KmInit = load('.\Results\NMF_Similarity\K_MeansInit\NMFResults_201803062210.mat',...
                'kFeat','mDC','stdDC','SpatialModules','ActCoeff');

%% Create plot to show reconstruction a data matrix
iN = 3; % 25% noise level
iC = 2; % 20% Non-Coding Patterns introduced

%Re-Create the whole trial-concatenated matrix 
X = zeros(nNeurons,nTrials*nBins);
groups_X = zeros(nTrials*nBins,1);
offset = 0;
for iStim = 1:nStimuli
    for iTrial = 1:nTrials
        %Trial-Concatenated Matrix
        X(:,offset+(1:nBins)) = data(iN,iC).counts{iStim}{iTrial};
        %Class labels
        groups_X(offset+(1:nBins),1) = iStim;
        %Update offset for data matrix
        offset = offset + nBins;
    end
end

figure
imagesc(X)
xticks([])
yticks([])
caxis([5,19])

W = KmInit.SpatialModules{iN,iC};
H = KmInit.ActCoeff{iN,iC};
err = norm(X - W*H,'fro')^2/norm(X,'fro')^2;
figure
imagesc(W)
xticks([])
yticks([])
caxis([5,19])

figure
imagesc(H)
xticks([])
yticks([])
caxis([0,1])
colorbar

figure
imagesc(W*H)
xticks([])
yticks([])
caxis([5,19])

%% Look at the similarity score during x-validation between the 2 different
%initializations for a sample noise/non-coding level
iN = 2;
iC = 2;

F1 = dir('.\Results\NMF_Similarity\K_MeansInit\ResultsN*mat');
F2 = dir('.\Results\NMF_Similarity\No_K_Means\ResultsN*mat');
K_xval = load(fullfile('.\Results\NMF_Similarity\K_MeansInit\',F1(iC).name));
NoK_xval = load(fullfile('.\Results\NMF_Similarity\No_K_Means\',F2(iC).name));

%Similarity score for 25% noise level for K-Means initialization
xvNMF = K_xval.rrNMF_xv{iN,1};
SimilarityScore = similarity_analysis(xvNMF);

mSS = [];
stdSS = [];
maxFeat = 6;
figure
for iK = 1:maxFeat
    pos = SimilarityScore(iK,:) ~= 1 & ~isnan(SimilarityScore(iK,:));
    mSS(iK) = mean(SimilarityScore(iK,pos));
    stdSS(iK) = std(SimilarityScore(iK,pos));
    plot(iK,SimilarityScore(iK,pos),'.b','MarkerSize',10),hold on
end
plot(1:maxFeat,mSS,'-b','LineWidth',2),hold on

%Similarity score for 25% noise level for random initialization
xvNMF = NoK_xval.rrNMF_xv{iN,1};
SimilarityScore = similarity_analysis(xvNMF);

mSS = [];
stdSS = [];
maxFeat = 6;
for iK = 1:maxFeat
    pos = SimilarityScore(iK,:) ~= 1 & ~isnan(SimilarityScore(iK,:));
    mSS(iK) = mean(SimilarityScore(iK,pos));
    stdSS(iK) = std(SimilarityScore(iK,pos));
    plot(iK,SimilarityScore(iK,pos),'.r','MarkerSize',10),hold on
end
plot(1:maxFeat,mSS,'-r','LineWidth',2),hold on

%% Look at the similarity score between the main NMF runs


%% Compare decoding performance for Kmeans and random seed Initializations 
figure
subplot(1,2,1)
imagesc(KmInit.mDC)
title('Decoding Performance for 4 Patterns using Kmeans Initialization');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([1,100])
colormap jet

subplot(1,2,2)
imagesc(NoInit.mDC)
title('Decoding Performance for 4 Patterns using Random Initialization');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:10)
yticks(1:1:10)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([1,100])
colormap jet

%Just plot the difference
figure
imagesc(KmInit.mDC-NoInit.mDC)
title('Difference in Decoding Performance using Kmeans vs Random Seed Initializations');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:10)
yticks(1:1:10)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([-5,5])
colormap jet