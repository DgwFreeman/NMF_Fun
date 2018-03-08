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
nNeurons = 20;
nTrials = 20;
nBins = 100;
nPatterns = 4;
nStimuli = 4;

%Supervised or Unsupervised X-validation?
Opt = struct;
Opt.SV = 0;
Opt.nXVAL = 10;     %# of x-validation runs

%Gaussian noise to add to each pattern as a % of the sigma_rate
noise = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3];
nNoise = length(noise);

%Fraction of non-coding patterns into stimulus presentation
fCoding = 0:0.1:0.9;
nCoding = length(fCoding);
nSessions = nCoding + nNoise;

%Create matrix for synthetic data
counts = cell(nStimuli,1);
trials = cell(nTrials,1);

%% Create Synthetic Data
% What is a reasonable mean firing rate and its standard deviation?
mean_rate = 10;
sigma_rate = 5;
SNR = mean_rate./(noise*sigma_rate);

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
if Opt.SV == 1
    %For Supervised Case, perform the leave-1-out cross-validation
    %Random permutation of trials
    pTrials = randperm(nTrials);
    
    % Total number of training samples
    n_e_train = nStimuli*(nTrials-1);
    % Total number of test samples
    n_e_test = nStimuli;
    
    %Start index at end of vector of trial indices to chose from
    TestIndex = nTrials;
    %Number of x-validation iterations
    Opt.nXVAL = nTrials;
else
    %For Unsupervised Case, separate data 50/50
    %Create nXVAL different combinations of the data split
    Opt.nXVAL = 10;
    ind_train = zeros(Opt.nXVAL,nTrials/2);
    ind_test = zeros(Opt.nXVAL,nTrials/2);
    for iXV = 1:Opt.nXVAL
        p = randperm(nTrials);
        %Training indices
        ind_train(iXV,:) = p(1:ceil(nTrials/2));
        %Test indices
        ind_test(iXV,:) = p((ceil(nTrials/2)+1):end);
    end
    % Total number of training samples
    n_e_train = nStimuli*size(ind_train,2);
    % Total number of test samples
    n_e_test = nStimuli*size(ind_train,2);
end
%% Loop over different noise conditions and calculate performance for each
%Preallocate for parallel processing
SpatialModules = cell(nNoise,nCoding);
ActCoeff = cell(nNoise,nCoding);
kFeat = zeros(nNoise,nCoding);
mDC = zeros(nNoise,nCoding);
stdDC = zeros(nNoise,nCoding);

rrNMF_W = cell(nNoise,1);
rrNMF_H = cell(nNoise,1);
rrNMF_xv = cell(nNoise,1);

%Save Data used for analysis
d = clock;
datastr = sprintf('./SimulatedData_%u%.2u%.2u%.2u%.2u.mat',d(1:5));
save(datastr);

% fprintf('Loading Data...\n');
% load('C:\Users\Freeman\Documents\GitHub\NMF_Fun\Results\KmeansInit\ExampleData_201802161635.mat','data');
% poolobj = parpool(10);

for iC = 1:nCoding
    fprintf('%u%% non-coding patterns introduced...\n',int16(fCoding(iC)*100));
    tStart = tic;
    for iN = 1:nNoise
        fprintf('\t Concatenating data for %u%% noise level...\n',int16(noise(iN)*100));
        % Build overall training and test matrices
        X_train = zeros(nNeurons,n_e_train*nBins);
        X_test = zeros(nNeurons,n_e_test*nBins);
        groups_train = zeros(n_e_train*nBins,1);
        groups_test = zeros(n_e_test*nBins,1);
        
        %% Cross Validation 
        DCperf = zeros(Opt.nXVAL,1);
        K_cv = zeros(Opt.nXVAL,1);
        minSQE = zeros(Opt.nXVAL,1);
        xvNMF = cell(Opt.nXVAL,3);
        
        %Loop over x-validation runs
        for iXV = 1:Opt.nXVAL
            offset_train = 0;
            offset_test = 0;
            
            %Loop over the different stiumuli & trials to create matrices
            for iStim = 1:nStimuli
                if Opt.SV == 1
                    %Supervised x-validation
                    for iTrial = 1:nTrials
                        if iTrial ~= TestIndex
                            X_train(:,offset_train+(1:nBins)) = data(iN,iC).counts{iStim}{pTrials(iTrial)};
                            %Training Class labels
                            groups_train(offset_train+(1:nBins),1) = iStim;
                            %Update offset for training matrix
                            offset_train = offset_train + nBins;
                        end
                    end
                    
                    X_test(:,offset_test+(1:nBins)) = data(iN,iC).counts{iStim}{pTrials(TestIndex)};
                    %Test class labels
                    groups_test(offset_test+(1:nBins),1) = iStim;
                    
                    %Update offset for test matrix
                    offset_test = offset_test + nBins;
                    %Update which trial is used for testing
                    TestIndex = TestIndex - 1;
                else
                    %Unsupervised x-validation, separate the training/test
                    for iTrial = 1:length(ind_train)
                        X_train(:,offset_train+(1:nBins)) = data(iN,iC).counts{iStim}{ind_train(iXV,iTrial)};
                        groups_train(offset_train+(1:nBins),1) = iStim;
                        offset_train = offset_train + nBins;
                    end
                    for iTrial = 1:length(ind_test)
                        X_test(:,offset_test+(1:nBins)) = data(iN,iC).counts{iStim}{ind_test(iXV,iTrial)};
                        groups_test(offset_test+(1:nBins),1) = iStim;
                        offset_test = offset_test + nBins;
                    end
                end
            end
            
            %% Select the Optimal Number of components, k, using the either
            % the unsupervised SQE formulation or similarity score, or the
            % supervised decoding performance
            [DCperf(iXV),minSQE(iXV), K_cv(iXV), xvNMF{iXV,1}] = select_k(X_train,groups_train,X_test,groups_test,Opt);

            %Save matricces used for xval for later similarity analysis
            xvNMF{iXV,2} = X_train;
            xvNMF{iXV,3} = X_test;
        end
        
        %Save X-Validation NMF runs for future analysis
        rrNMF_xv{iN,1} = xvNMF;
        
        %Out of all of the "Leave 1 out" iterations, which one resulted in
        %the smallest reconstruction error out of the median values of K
        pos = find(K_cv == median(K_cv));
        [~,iK] = min(minSQE(pos));
        kFeat(iN,iC) = K_cv(pos(iK));
        
        %% Now that we've determined the number of components to extract
        %create the whole trial-concatenated matrix to input into the NMF
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
        
        %% Decompose the data now using the k value found in the cross-validation algorithm
        % Run NMF multiple times to average decoding performance
        nNMFruns = 10;
        %Calculate the squared error of each run 
        SQE = zeros(nNMFruns,1);
        %Compare Features and Coefficients between runs of the nmf
        rrFeatures = cell(nNMFruns,1);
        rrActCoeff = cell(nNMFruns,1);
        %Compare the decoding performance between each run
        rrDC = zeros(nNMFruns,1);

        for indy = 1:nNMFruns
            %Decompose the trial-concatenated data
            [W,H,err] = nmf(X,kFeat(iN,iC));      
      
            %Calculate Squared Error
            SQE(indy,1) = norm(X-W*H,'fro')^2/norm(X,'fro')^2;
            
            %Process activation coefficients for classification
            predictors = H';
            [cc,~] = ldacc(predictors,groups_X);
            
            %Save factorized representation of data
            rrFeatures{indy,1} = W;
            rrActCoeff{indy,1} = H;

            %Save Decoding Performance
            rrDC(indy,1) = cc;
        end
        rrNMF_W{iN,1} = rrFeatures;
        rrNMF_H{iN,1} = rrActCoeff;      
        
        %Save factorized representation of data with the smallest SE
        [~,indy] = min(SQE);       
        SpatialModules{iN,iC} = rrFeatures{indy,1};
        ActCoeff{iN,iC} = rrActCoeff{indy,1};
        
        %Take the average decoding performance out of the reruns
        mDC(iN,iC) = mean(rrDC);
        stdDC(iN,iC) = std(rrDC);
    end
    
    %Save data as we go along
    d = clock;
    datastr = sprintf('./ResultsNClevel%u_%u%.2u%.2u%.2u%.2u.mat',int16(fCoding(iC)*100),d(1:5));
    save(datastr,'rrNMF_W','rrNMF_H','rrNMF_xv','kFeat','SpatialModules','ActCoeff','mDC','stdDC');
    
    tElapsed = toc(tStart);
    fprintf('Time Elapsed for %u%% Non-Coding Level: %3.3f mins\n',int16(fCoding(iC)*100), tElapsed/60);
end

%% Clean up workspace & Save Results
delete(poolobj);
fprintf('Done with NMF analysis\n');
d = clock;
datastr = sprintf('./NMFResults_%u%.2u%.2u%.2u%.2u.mat',d(1:5));
save(datastr);
