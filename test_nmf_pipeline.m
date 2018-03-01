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

% %Create matrix for synthetic data
% counts = cell(nStimuli,1);
% trials = cell(nTrials,1);
% 
% %% Create Synthetic Data
% % What is a reasonable mean firing rate and its standard deviation?
% mean_rate = 10;
% sigma_rate = 5;
% 
% %Parameters of log-normal distribution
% mu = log(mean_rate^2/sqrt(sigma_rate + mean_rate^2));
% sigma = sqrt(log(sigma_rate/mean_rate^2 + 1));
% 
% %Draw 4 patterns of the firing rates of 100 neurons from the log-normal distribution
% for ii = 1:nPatterns
%     Patterns{ii,1} = lognrnd(mu,sigma,nNeurons,1);
% end
% 
% CountMatrix = zeros(nNeurons,nBins);
% data = repmat(struct,nNoise,nCoding);
% for iN = 1:nNoise
%     for iC = 1:nCoding
%         fprintf('Creating simulated data for %u%% noise & %u%% non-coding patterns...\n',int16(noise(iN)*100),int16(fCoding(iC)*100));
%         for iStim = 1:nStimuli
%             %Which Pattern to select
%             iP = mod(iStim-1,4) + 1;
%             
%             %Base Firing Rate Pattern
%             RatePattern = Patterns{iP};
%             
%             %For each neuron and time step, add gaussian noise
%             for iTrial = 1:nTrials
%                 %Additive Noise same across all patterns for each trial
%                 CountMatrix = repmat(RatePattern,1,nBins) + noise(iN)*sigma_rate*randn(nNeurons,nBins);
%                 
%                 %Randomly insert "non-coding" patterns into the trial based off
%                 %the fraction of coding patterns variable fCoding
%                 nncBins = int8(fCoding(iC)*nBins);
%                 p = randperm(nBins);
%                 ind_nc = p(1:nncBins);
%                 for iNC = 1:length(ind_nc)
%                     NonCodingPattern = lognrnd(mu,sigma,nNeurons,1) + noise(iN)*sigma_rate*randn(nNeurons,1);
%                     CountMatrix(:,ind_nc(iNC)) = NonCodingPattern;
%                 end
%                 
%                 %Set negative values to 0
%                 negPos = CountMatrix < 0;
%                 CountMatrix(negPos) = 0;
%                 
%                 trials{iTrial,1} = CountMatrix;
%             end
%             counts{iStim,1} = trials;
%         end
%         
%         %Save data
%         data(iN,iC).counts = counts;
%         data(iN,iC).noise = noise(iN);
%         data(iN,iC).fCoding = fCoding(iC);
%         
%     end
%     
% end

%% Randomly separate data into training set and test set
%For Unsupervised Case, separate data 50/50
%Create nXVAL different combinations of the data split
nXVAL = 10;
ind_train = zeros(nXVAL,nTrials/2);
ind_test = zeros(nXVAL,nTrials/2);
for iXV = 1:nXVAL
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

%For Supervised Case, perform the leave-1-out cross-validation
% % Random permutation of trials
% pTrials = randperm(nTrials);
%
% % Total number of training samples
% n_e_train = nStimuli*(length(pTrials)-1);
% % Total number of test samples
% n_e_test = nStimuli;

%% Loop over different noise conditions and calculate performance for each
%Preallocate for parallel processing
SpatialModules = cell(nNoise,nCoding);
TestCoeff = cell(nNoise,nCoding);
TrainCoeff = cell(nNoise,nCoding);
kFeat = zeros(nNoise,nCoding);
dctr = zeros(nNoise,nCoding);
dcte = zeros(nNoise,nCoding);
std_dctr = zeros(nNoise,nCoding);
std_dcte = zeros(nNoise,nCoding);
mean_fCorr = cell(nNoise,nCoding);
std_fCorr = cell(nNoise,nCoding); 
mean_tcCorr = cell(nNoise,nCoding);
std_tcCorr = cell(nNoise,nCoding);

%Save Data used for analysis
% d = clock;
% datastr = sprintf('./ExampleData_%u%.2u%.2u%.2u%.2u.mat',d(1:5));
% save(datastr);

fprintf('Loading Data...\n');
load('C:\Users\Freeman\Documents\GitHub\NMF_Fun\Results\KmeansInit\ExampleData_201802161635.mat','data');
% poolobj = parpool(5);

for iC = 1:nCoding
    fprintf('\t %u%% non-coding patterns introduced...\n',int16(fCoding(iC)*100));
    tStart = tic;
    for iN = 3:nNoise
        fprintf('Concatenating data for %u%% noise level...\n',int16(noise(iN)*100));
        % Build overall training and test matrices
        X_train = zeros(nNeurons,n_e_train*nBins);
        X_test = zeros(nNeurons,n_e_test*nBins);
        groups_train = zeros(n_e_train*nBins,1);
        groups_test = zeros(n_e_test*nBins,1);
        
        %% Leave 1 Out cross validation
        %Loop over each "leave 1 out" interation of the cross-validation
        %algorithm to obtain the best k based on the decoding performance
        %Start index at end of vector of trial indices to chose from
%         TestIndex = length(pTrials);
%         DCperf = zeros(length(pTrials),1);
%         K_cv = zeros(length(pTrials),1);
%         minSQE = zeros(length(pTrials),1);
        
        %% Random sub-sampling validation
        DCperf = zeros(nXVAL,1);
        K_cv = zeros(nXVAL,1);
        minSQE = zeros(nXVAL,1);
        for iXV = 1:nXVAL
            offset_train = 0;
            offset_test = 0;
            
            %Loop over the different stiumuli & trials to create matrices
            for iStim = 1:nStimuli
                %Supervised xval 
%                 for iTrial = 1:nTrials
%                     if iTrial ~= TestIndex
%                         X_train(:,offset_train+(1:nBins)) = data(iN,iC).counts{iStim}{pTrials(iTrial)};
%                         %Training Class labels
%                         groups_train(offset_train+(1:nBins),1) = iStim;
%                         %Update offset for training matrix
%                         offset_train = offset_train + nBins;
%                     end
%                 end
%                 
%                 X_test(:,offset_test+(1:nBins)) = data(iN,iC).counts{iStim}{pTrials(TestIndex)};
%                 %Test class labels
%                 groups_test(offset_test+(1:nBins),1) = iStim;
%                 
%                 %Update offset for test matrix
%                 offset_test = offset_test + nBins;
%                 %Update which trial is used for testing
%                 TestIndex = TestIndex - 1;
                
                %Unsupervised cross-validation, separate the training/test
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
            
            %% Select the Optimal Number of components, k, using the 
            % unsupervised SQE formulation
            [DCperf(iXV),minSQE(iXV), K_cv(iXV)] = select_k(X_train,groups_train,X_test,groups_test);
            
        end
        
        %Out of all of the "Leave 1 out" iterations, which one resulted in
        %the smallest reconstruction error out of the median values of K
        pos = find(K_cv == median(K_cv));
        [SQE,iK] = min(minSQE(pos));
        kFeat(iN,iC) = K_cv(pos(iK));
        TestIndex = pos(iK);
        
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
            SQE(indy) = norm(X-W*H,'fro')^2/norm(X,'fro')^2;
            
            %Process activation coefficients for classification
            predictors = H';
            [cc,~] = ldacc(predictors,groups_X);
            
            %Save factorized representation of data
            rrFeatures{indy,1} = W;
            rrActCoeff{indy,1} = H;

            %Save Decoding Performance
            rrDC(indy,1) = cc_train;
        end
        
        %Save factorized representation of data with the smallest SE
        [~,indy] = min(sqerr_te);       
        SpatialModules{iN,iC} = rrFeatures{indy,1};
        TestCoeff{iN,iC} = rrTestCoeff{indy,1};
        TrainCoeff{iN,iC} = rrTrainCoeff{indy,1};
        
        %Take the average decoding performance out of the reruns
        dctr(iN,iC) = mean(rr_ctr);
        dcte(iN,iC) = mean(rr_cte);
        
        std_dctr(iN,iC) = std(rr_ctr);
        std_dcte(iN,iC) = std(rr_cte);
        
        %% Determine how correlated each NMF run is with each other
        featureCorr = zeros(kFeat(iN,iC),kFeat(iN,iC));
        fCorr = zeros(kFeat(iN,iC),1);
        iCorr = zeros(kFeat(iN,iC),1);
        tcCorr = zeros(kFeat(iN,iC),1);
        
        %Mean and standard deviation of the features between runs
        m_fCorr = NaN(nNMFruns);
        s_fCorr = NaN(nNMFruns);
        m_tcCorr = NaN(nNMFruns);
        s_tcCorr = NaN(nNMFruns);
        for ii = 1:nNMFruns
            aFeatures = rrFeatures{ii};
            aTestCoeff = rrTestCoeff{ii};
            for jj = 1:nNMFruns
                bFeatures = rrFeatures{jj};
                bTestCoeff = rrTestCoeff{jj};
                
                %Find the features between iFeatures & jFeatures that 
                %correspond to the maximum correlation coefficient
                for iK = 1:kFeat(iN,iC)
                    featureCorr = [];
                    for jK = 1:kFeat(iN,iC)
                        featureCorr = [featureCorr, corr2(aFeatures(:,iK),bFeatures(:,jK))];
                    end
                    %Calculate the max correlation of each feature in aFeatures with
                    %that of each feature in bFeatures & the corresponding index
                    [fCorr(iK), iCorr(iK)] = max(featureCorr);
                    bFeatures(:,iCorr(iK)) = 0;
                    tcCorr(iK) = corr2(aTestCoeff(iK,:),bTestCoeff(iCorr(iK),:));
                end
 
                m_fCorr(ii,jj) = mean(fCorr);
                s_fCorr(ii,jj) = std(fCorr);
                m_tcCorr(ii,jj) = mean(tcCorr);
                s_tcCorr(ii,jj) = std(tcCorr);
            end
        end

       mean_fCorr{iN,iC} = m_fCorr;
       std_fCorr{iN,iC} = s_fCorr;
       mean_tcCorr{iN,iC} = m_tcCorr;
       std_tcCorr{iN,iC} = s_tcCorr;
        
    end
    
    %Save data as we go along
    d = clock;
    datastr = sprintf('./ResultsNClevel%u_%u%.2u%.2u%.2u%.2u.mat',int16(fCoding(iC)*100),d(1:5));
    save(datastr,'mean_fCorr','std_fCorr','mean_tcCorr','std_tcCorr','SpatialModules','TestCoeff','TrainCoeff','dctr','dcte','std_dctr','std_dcte');
    
    tElapsed = toc(tStart);
    fprintf('Time Elapsed for %u%% Non-Coding Level: %3.3f mins\n',int16(fCoding(iC)*100), tElapsed/60);
end

%% Clean up workspace & Save Results
delete(poolobj);
fprintf('Done with NMF analysis\n');
d = clock;
datastr = sprintf('./NMFResults_%u%.2u%.2u%.2u%.2u.mat',d(1:5));
save(datastr,'mean_fCorr','std_fCorr','mean_tcCorr','std_tcCorr','SpatialModules','TestCoeff','TrainCoeff','dctr','dcte','std_dctr','std_dcte','data','kFeat');
