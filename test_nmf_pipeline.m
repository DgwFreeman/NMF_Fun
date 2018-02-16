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
% Leave 1 out cross-validation
% Random permutation of trials
pTrials = randperm(nTrials);

% % Training indices
% ind_train = p(1:ceil(nTrials/2));
% % Test indices
% ind_test = p((ceil(nTrials/2)+1):end);

% Total number of training samples
n_e_train = nStimuli*(length(pTrials)-1);
% Total number of test samples
n_e_test = nStimuli;

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
d = clock;
datastr = sprintf('./ExampleData_%u%.2u%.2u%.2u%.2u.mat',d(1:5));
save(datastr);

poolobj = parpool(15);

for iC = 1:nCoding
    fprintf('\t %u%% non-coding patterns introduced...\n',int16(fCoding(iC)*100));
    tStart = tic;
    parfor iN = 1:nNoise
        fprintf('Concatenating data for %u%% noise level...\n',int16(noise(iN)*100));
        % Build overall data matrices
        X_train = zeros(nNeurons,n_e_train*nBins);
        X_test = zeros(nNeurons,n_e_test*nBins);
        groups_train = zeros(n_e_train*nBins,1);
        groups_test = zeros(n_e_test*nBins,1);
        
        %% Leave 1 Out cross validation
        %Loop over each "leave 1 out" interation of the cross-validation
        %algorithm to obtain the best k based on the decoding performance
        %Start index at end of vector of trial indices to chose from
        TestIndex = length(pTrials);
        DCperf = zeros(length(pTrials),1);
        K_cv = zeros(length(pTrials),1);
        for iCross = 1:length(pTrials)
            offset_train = 0;
            offset_test = 0;
            
            %Loop over the different stiumuli & trials to create matrices
            for iStim = 1:nStimuli
                for iTrial = 1:length(pTrials)
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
            end
            %Find Optimal number of spatial modules for this particular
            %training/test set combo
            [DCperf(iCross,1), K_cv(iCross,1)] = select_k(X_train,groups_train,X_test,groups_test);
            
            %Update which trial is used for testing
            TestIndex = TestIndex - 1;
        end
        
        %Out of all of the "Leave 1 out" iterations, which one resulted in
        %the best decoding performance?
        [DCmax,iK] = max(DCperf(:));
        kFeat(iN,iC) = K_cv(iK);
        TestIndex = iK;
        
        %Re-create the training and test matrices that resulted in the best
        %decoding performance
        offset_train = 0;
        offset_test = 0;
        for iStim = 1:nStimuli
            for iTrial = 1:length(pTrials)
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
        end
        
        %% Decompose the data now using the k value found in the cross-validation algorithm
        % Run NMF multiple times to average decoding performance
        nNMFruns = 10;
        %Calculate the squared error of each run 
        sqerr_tr = zeros(nNMFruns,1);
        sqerr_te = zeros(nNMFruns,1);
        %Compare Features and Coefficients between runs of the nmf
        rrFeatures = cell(nNMFruns,1);
        rrTestCoeff = cell(nNMFruns,1);
        rrTrainCoeff = cell(nNMFruns,1);
        %Compare the decoding performance between each run
        rr_ctr = zeros(nNMFruns,1);
        rr_cte = zeros(nNMFruns,1);
        
        for indy = 1:nNMFruns
            %Decompose training set
            [W_train,H_train,err] = nmf(X_train,kFeat(iN,iC));
            
            %Obtain test set activation coefficients for given modules
            [~,H_test,~] = nmf(X_test,kFeat(iN,iC),W_train);
            
            %Decompose training set with VSMF function
%             feMethod = 'vsmf';
%             max_iter=10000;
%             err_tol=1e-12;
%             %NMF Options
%             Opt_VSMF = struct('iter',max_iter,'tof',err_tol,'dis',false,...
%                 'alpha2',0.02,'alpha1',0.02,'lambda2',0.02,'lambda1',0.02,...
%                 't1',true,'t2',true,'kernelizeAY',0,'feMethod',feMethod);
%             
%             %Run the training set through the VSMF algorithm
%             [W_train,H_train,WtW_train] = vsmf(X_train,kFeat(iN,iC),Opt_VSMF);
%             
%             %Save Training results
%             TrainingOutput = cell(4,1);
%             TrainingOutput{1} = W_train;
%             TrainingOutput{2} = H_train;
%             TrainingOutput{3} = WtW_train;
%             TrainingOutput{4} = X_train;
% 
%             %Run the test set through the VSMF algorithm
%             [W_test,H_test,WtW_test] = vsmf(X_test,kFeat(iN,iC),Opt_VSMF,TrainingOutput);
                       
            %Calculate Squared Error
            sqerr_tr(indy) = norm((X_train - W_train*H_train).^2,'fro');
            sqerr_te(indy) = norm((X_test - W_train*H_test).^2,'fro');
            
            %Process activation coefficients for classification
            predictors_train = H_train';
            predictors_test = H_test';
            [cc_train,cc_test] = ldacc(predictors_train,groups_train,predictors_test,groups_test);
            
            %Save factorized representation of data
            rrFeatures{indy,1} = W_train;
            rrTestCoeff{indy,1} = H_test;
            rrTrainCoeff{indy,1} = H_train;
            
            %Save Decoding Performance
            rr_ctr(indy,1) = cc_train;
            rr_cte(indy,1) = cc_test;
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
delete(poolobj);
fprintf('Done with NMF analysis\n');
d = clock;
datastr = sprintf('./NMFResults_%u%.2u%.2u%.2u%.2u.mat',d(1:5));
save(datastr,'mean_fCorr','std_fCorr','mean_tcCorr','std_tcCorr','SpatialModules','TestCoeff','TrainCoeff','dctr','dcte','std_dctr','std_dcte','data','kFeat');
