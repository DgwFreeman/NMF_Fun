clear all
plot_figs = 0;
nNeurons = 20;
nTrials = 20;
nBins = 100;
nCodingPatterns = 4;
nNonCodingPatterns = 12;
nPatterns = nCodingPatterns + nNonCodingPatterns;
nStimuli = 4;
kMax = 50;

%Gaussian noise to add to each pattern as a % of the sigma_rate
noise = 0:0.1:1;
noise = [noise, [1.5, 2, 2.5, 3]];
nNoise = length(noise);

%Fraction of non-coding patterns into stimulus presentation
fCoding = 0:0.06:0.84;
nCoding = length(fCoding);
nSessions = nCoding + nNoise;

%Create matrix for synthetic data
counts = cell(nStimuli,1);
trials = cell(nTrials,1);

%% Create Synthetic Data
% What is a reasonable mean firing rate and its standard deviation?
mean_rate = 10;
sigma_rate = 5;

%Parameters of log-normal distribution
mu = log(mean_rate^2/sqrt(sigma_rate + mean_rate^2));
sigma = sqrt(log(sigma_rate/mean_rate^2 + 1));

%Draw 4 coding patterns of the firing rates from the log-normal dist
CodingPatterns = cell(nCodingPatterns,1);
for ii = 1:nCodingPatterns
    CodingPatterns{ii,1} = lognrnd(mu,sigma,nNeurons,1);
end

%Draw 12 non-coding patterns of the firing rates from the log-normal dist
NonCodingPatterns = cell(nCodingPatterns,1);
for ii = 1:nNonCodingPatterns
    NonCodingPatterns{ii,1} = lognrnd(mu,sigma,nNeurons,1);
end

%% Similarity Analysis
iN = 2;

Extr_Features = cell(kMax,1);
for iK = 1:kMax
    %Create fake NMF results of iK extracted features
    NMF_Features = zeros(nNeurons,iK);
    
    %Select non-coding patterns randomly
    for iP = 1:iK
        if iP <= nCodingPatterns
            %Coding Firing Rate Pattern
            NMF_Features(:,iP) = CodingPatterns{iP} + noise(iN)*sigma_rate*randn(nNeurons,1);
        else
            %Non-Coding Firing Rate Pattern
            indy = randperm(nNonCodingPatterns,1);
            NMF_Features(:,iP) = NonCodingPatterns{indy}+ noise(iN)*sigma_rate*randn(nNeurons,1);
        end
    end
    
    %Save fake NMF results for subsequent similarity analysis
    Extr_Features{iK,1} = NMF_Features;
end

%Compare features extracted for iK and iK+1
fCorrAll = [];
fCorrBM = [];
figure
for iK = 1:kMax-1
    iK_Features = Extr_Features{iK,1};
    iKplus1_Features = Extr_Features{iK+1,1};
    
    featureCorr = zeros(iK,iK+1);
    fCorr = zeros(iK,1);
    iCorr = zeros(iK,1);
    %Find the features between iK_Features & iKplus1_Features that
    %correspond to the maximum correlation coefficient
    for ii = 1:iK
        for jj = 1:iK+1
            featureCorr(ii,jj) = corr2(iK_Features(:,ii),iKplus1_Features(:,jj));
        end
        %Calculate the max correlation of each feature in iK_Features with
        %that of each feature in iKplus1_Features & the corresponding index
        
        [fCorr(ii), iCorr(ii)] = max(featureCorr(ii,:));
       
         iKplus1_Features(:,iCorr(ii)) = 0;
    end
    fCorrAll = [fCorrAll; featureCorr(:)];
    fCorrBM = [fCorrBM; fCorr(:)];
%     histogram(fCorr);
    
    SimilarityIndex(iK,1) = mean(fCorr);
    
    
end
    

