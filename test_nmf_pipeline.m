%% test_nmf_pipeline.m
% This is a script to run synthetic data through the NMF pipeline to
% validate the algorithm is working as expected. 4 "patterns" of firing
% rates will be generated from a log-normal distribution for 100 neurons.
% For each of trial, each time step will be a jittered pattern of the
% firing rate vector. There will be 20 trials per pattern and 30 time bins
% per trial of 10ms each. The noise added will be gaussian for now.
%
% Author: D Wyrick
% Date: 1/24/18
%% Parameters
clear all
plot_figs = 0;
nNeurons = 20;
nTrials = 20; 
nBins = 100; 
nPatterns = 4;
nStimuli = 4;

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

%Draw 4 patterns of the firing rates of 100 neurons from the log-normal distribution
for ii = 1:nPatterns
    Patterns{ii,1} = lognrnd(mu,sigma,nNeurons,1);
end

CountMatrix = zeros(nNeurons,nBins);
data = repmat(struct,nNoise,nCoding);
for iN = 1:nNoise
    for iC = 1:nCoding
        fprintf('Creating simulated data for %u%% noise & %u%% non-coding patterns...\n',int16(noise(iN)*100),int16(fCoding(iC)*100));
        for iStim = 1:nStimuli
            %Which Pattern to select
            iP = mod(iStim-1,4) + 1;
            
            %Base Firing Rate Pattern
            RatePattern = Patterns{iP};
            
            %For each neuron and time step, add gaussian noise
            for iTrial = 1:nTrials
                %Additive Noise same across all patterns for each trial
                CountMatrix = repmat(RatePattern,1,nBins) + noise(iN)*sigma_rate*randn(nNeurons,nBins);
                
                %Randomly insert "non-coding" patterns into the trial based off
                %the fraction of coding patterns variable fCoding
                nncBins = int8(fCoding(iC)*nBins);
                p = randperm(nBins);
                ind_nc = p(1:nncBins);
                for iNC = 1:length(ind_nc)
                    NonCodingPattern = lognrnd(mu,sigma,nNeurons,1) + noise(iN)*sigma_rate*randn(nNeurons,1);
                    CountMatrix(:,ind_nc(iNC)) = NonCodingPattern;
                end
                
                %Set negative values to 0
                negPos = CountMatrix < 0;
                CountMatrix(negPos) = 0;
                
                trials{iTrial,1} = CountMatrix;
            end
            counts{iStim,1} = trials;
        end
        
        %Save data
        data(iN,iC).counts = counts;
        data(iN,iC).noise = noise(iN);
        data(iN,iC).fCoding = fCoding(iC);

    end
    
end

%% Randomly separate data into training set and test set
% Random permutation of trials
p = randperm(nTrials); 

% Training indices
ind_train = p(1:ceil(nTrials/2));
% Test indices
ind_test = p((ceil(nTrials/2)+1):end);

% Total number of training samples
n_e_train = nStimuli*length(ind_train);
% Total number of test samples
n_e_test = nStimuli*length(ind_test);

%% Loop over different noise conditions and calculate performance for each
%Save factorized representation
SpatialModules = cell(nSessions,1);
TestCoeff = cell(nSessions,1);
TrainCoeff = cell(nSessions,1);

% load('Dataset_20180205.mat');
for iN = 1:nNoise
    fprintf('Concatenating data for %u%% noise level...\n',int16(noise(iN)*100));
    tStart = tic;
    for iC = 1:nCoding
        fprintf('\t %u%% non-coding patterns introduced...\n',int16(fCoding(iC)*100));
        
        %% Build overall data matrices
        X_train = zeros(nNeurons,n_e_train*nBins);
        X_test = zeros(nNeurons,n_e_test*nBins);
        offset = 0;
        for iStim = 1:nStimuli
            for iTrial = 1:length(ind_train)
                X_train(:,offset+(1:nBins)) = data(iN,iC).counts{iStim}{ind_train(iTrial)};
                X_test(:,offset+(1:nBins)) = data(iN,iC).counts{iStim}{ind_test(iTrial)};
                
                % Training & Test class labels
                groups_train(offset+(1:nBins),1) = iStim;
                groups_test(offset+(1:nBins),1) = iStim;
                
                offset = offset + nBins;
            end
        end
        
        %Find Optimal number of spatial modules
        k = select_k(X_train,groups_train,n_e_train,X_test,groups_test,n_e_test);
        kVec(iN,iC) = k;
        
        % Decompose training set
        [W_train,H_train,err] = nmf(X_train,k);
        
        % Obtain test set activation coefficients for given modules
        [~,H_test,~] = nmf(X_test,k,W_train);
        
        %Save factorized representation of data
        SpatialModules{iN,iC} = W_train;
        TestCoeff{iN,iC} = H_test;
        TrainCoeff{iN,iC} = H_train;
        
        % Process activation coefficients for classification
        predictors_train = H_train';
        predictors_test = H_test';
        [cc_train,cc_test] = ldacc(predictors_train,groups_train,predictors_test,groups_test);
        
        ctr(iN,iC) = cc_train;
        cte(iN,iC) = cc_test;
        
        % feature extraction
        %     feMethod='nmf';
        %     optionFE.facts=3;
        %     [trainExtr,outTrain]=featureExtractionTrain(trainSet,[],trainClass,feMethod,optionFE);
        %     [testExtr,outTest]=featureExtrationTest(trainSet,testSet,outTrain);
        
        %% Optional Plotting
        if plot_figs == 1
            figure
            ss = sprintf('Noise Level: %u%% -- Non-Coding Percentage: %u%% -- DC Performance: %2.1f%%',int16(noise(iN)*100),int16(fCoding(iC)*100),cc_test);
            suptitle(ss);
            offset = 0;
            for iP = 1:nStimuli
                %Display one trial for each noisey pattern that was factorized
                NoisyPattern = X_test(:,offset+(1:nBins));
                
                subplot(4,3,(iP-1)*3+1)
                imagesc(NoisyPattern)
                ylabel('Neuron ID')
                title(sprintf('Pattern %u with noise',iP))
                colorbar
                caxis([8,18])
                if iP == 4, xlabel('Time (ms)');end
                
                %Display the data without noise
                Pattern_NoNoise = data(1).counts{iP}{ind_test(1)};
                
                subplot(4,3,(iP-1)*3+2)
                imagesc(Pattern_NoNoise)
                title('With no noise')
                colorbar
                caxis([8,18])
                if iP == 4, xlabel('Time (ms)');end
                
                %Get the activation coefficients for this particular sample
                %     P = Wi*Acal_train(:,:,(ii-1)*10+1)*Wb;
                P = SpatialModules{iN,iC}*TestCoeff{iN,iC};
                Pattern_NMF = P(:,offset+(1:nBins));
                
                subplot(4,3,(iP-1)*3+3)
                imagesc(Pattern_NMF)
                title('Factorized Representation')
                colorbar
                caxis([8,18])
                if iP == 4, xlabel('Time (ms)');end
                
                Err = norm(NoisyPattern-Pattern_NMF,'fro')^2;
                
                %shift offset to get the first trial of the next pattern in X_train/X_test
                offset = offset + nBins*nTrials/2;
            end
        end
            
    end
    tElapsed = toc(tStart);
    fprintf('Time Elapsed for %u%% Noise Level: %3.3f\n',int16(noise(iN)*100), tElapsed);
end

%% Troubleshooting / Plotting
% Compare noisy data with pristine data & factorized representation for
% first trial of the training set

cc = hsv(nNoise);
for iN = 1:nNoise
    fprintf('Plotting %u%% noise level...\n',int16(noise(iN)*100));
  
    for iC = 1:nCoding
        fprintf('\t %u%% non-coding patterns introduced...\n',int16(fCoding(iC)*100));
        
        %% Build overall data matrices
        X_train = zeros(nNeurons,n_e_train*nBins);
        X_test = zeros(nNeurons,n_e_test*nBins);
        offset = 0;
        for iStim = 1:nStimuli
            for iTrial = 1:length(ind_train)
                X_train(:,offset+(1:nBins)) = data(iN,iC).counts{iStim}{ind_train(iTrial)};
                X_test(:,offset+(1:nBins)) = data(iN,iC).counts{iStim}{ind_test(iTrial)};
                
                offset = offset + nBins;
            end
        end
        
        %Normalized Test Matrix and Factorized Matrix representation
        NoiseyPattern = X_test;
        P = SpatialModules{iN,iC}*TestCoeff{iN,iC};
        Pattern_NMF = P;
        for iT = 1:(nBins*nTrials*nPatterns/2)
            NoiseyPattern(:,iT) = NoiseyPattern(:,iT)/norm(NoiseyPattern(:,iT));
            Pattern_NMF(:,iT) = Pattern_NMF(:,iT)/norm(Pattern_NMF(:,iT));
        end
        
        ss = sprintf('Noise Level: %u%% -- Non-Coding Percentage: %u%%',int16(noise(iN)*100),int16(fCoding(iC)*100));
        figure
        suptitle(ss);
        
        subplot(1,3,1)
        imagesc(NoiseyPattern)
        colorbar
        caxis([0,1])
        title('Firing Rate Data')
        xlabel('Time(ms)')
        ylabel('Neuron ID')
        
        subplot(1,3,2)
        imagesc(Pattern_NMF)
        colorbar
        caxis([0,1])
        title('Factorized Representation')
        xlabel('Time(ms)')
        ylabel('Neuron ID')
        
            
        
    end
end

for ii = 1:20
    subplot(4,5,ii)
    imagesc(data(ii).counts{1,1}{1,1})
    xlabel('Time');ylabel('Neuron ID')
    title(sprintf('%2.2f %%',fCoding(ii)))
    colorbar
    caxis([8,18])
end

figure
plot(noise*100,cte,'.k','MarkerSize',15)
xlabel('Percent Additive Noise')
ylabel('Decoding Performance')
title('Decoding Performance for 4 Patterns');

figure
imagesc(cte)
title('Decoding Performance for 4 Patterns');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([1,100])
colormap jet


cc = hsv(nNoise);
tvec = 1:1:(nBins*nTrials*nPatterns/2);
for iN = 1:nNoise
    
    fprintf('Plotting %u%% noise level...\n',int16(noise(iN)*100));
    figure
    for iC = 1:nCoding
     
        Coeff = TestCoeff{iN,iC};
        for iM = 1:kVec(iN,iC)
            plot(tvec,Coeff(iM,:),'-','Color',cc(iC,:)),hold on
        end
        
    end
end

testBasis = SpatialModules{1, 4};
for i = 1:4
    tt = norm(testPatterns(:,i));
    testPatterns(:,i) = testPatterns(:,i)/tt;
    
    tt = norm(testBasis(:,i));
    testBasis(:,i) = testBasis(:,i)/tt;
    
end
figure;
subplot(2,1,1);
imagesc(testPatterns);
colorbar
caxis([0,1])

subplot(2,1,2);
imagesc(testBasis)
colorbar
caxis([0,1])

%% Show factorized matrices
figure
subplot(k,3,1:3:k*3)
imagesc(X_test)

subplot(k,3,2:3:k*3)
imagesc(W_train)

tvec = 1:1:length(H_Test);
for iP = 1:k
    subplot(k,3,iP*3)
    plot(tvec,H_Test(iP,:),'-','LineWidth',2)
end
    
