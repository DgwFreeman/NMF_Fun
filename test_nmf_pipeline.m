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
nNeurons = 100;
nTrials = 20; 
nBins = 30; 
nPatterns = 4;
nStimuli = 4;
nSessions = 4;

%Gaussian noise to add to each pattern as a % of the sigma_rate
noise = [0, 0.1, 0.25, 0.5];     

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

%  trials    - Cell array (length #trials) with spike count matrices of
%              size (#cells x #bins_per_trial)
CountMatrix = zeros(nNeurons,nBins);
for iSess=1:nSessions
    for iStim = 1:nStimuli
        %Which Pattern to select
        iP = mod(iStim-1,4) + 1;
        
        %Base Firing Rate Pattern
        RatePattern = Patterns{iP};
        
        %For each neuron and time step, add gaussian noise
        for iTrial = 1:nTrials
            %Additive Noise same across all patterns for each trial
            CountMatrix = (repmat(RatePattern,1,nBins) + noise(iSess)*sigma_rate*randn(nNeurons,nBins));
            
            %Set negative values to 0
            negPos = CountMatrix < 0;
            CountMatrix(negPos) = 0;
            
            trials{iTrial,1} = CountMatrix;
        end
        counts{iStim,1} = trials;
    end
    data(iSess).counts = counts;
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
for iSess = 4:nSessions
    fprintf('Concatenating data for %d percent noise level...\n',noise(iSess)*100);
    
    %% Build overall data matrices
    X_train = zeros(nNeurons,n_e_train*nBins);
    X_test = zeros(nNeurons,n_e_test*nBins);
    offset = 0;
    for iStim = 1:nStimuli
        for iTrial = 1:length(ind_train)
            X_train(:,offset+(1:nBins)) = data(iSess).counts{iStim}{ind_train(iTrial)};
            X_test(:,offset+(1:nBins)) = data(iSess).counts{iStim}{ind_test(iTrial)};
            offset = offset + nBins;
        end
    end
    
    % Training & Test class labels
    groups_train = ceil((1:n_e_train)' / length(ind_train));
    groups_test = ceil((1:n_e_test)' / length(ind_train));
    
    %% Apply space-by-time NMF to factorize data
    fprintf('Optimizing # of spatial modules with 1 temporal module\n');
    
    % Find optimal numbers of temporal and spatial modules
    [n_tm,n_sm] = select_n_tm_n_sm(X_train,groups_train,n_e_train,X_test,groups_test,n_e_test);
    
    % Obtain temporal and spatial modules from training set
    [Acal_train,Wi,Wb] = sbtnmf(X_train,n_tm,n_sm,n_e_train);
    
    % Obtain activation coefficients from test set for given modules
    Acal_test = sbtnmf(X_test,n_tm,n_sm,n_e_test,Wi,Wb);
    
    % Process activation coefficients for classification
    predictors_train = zeros(n_e_train,n_tm*n_sm);
    for i = 1:n_e_train
        predictors_train(i,:) = reshape(Acal_train(:,:,i),1,n_tm*n_sm);
    end
    predictors_test = zeros(n_e_test,n_tm*n_sm);
    for i = 1:n_e_test
        predictors_test(i,:) = reshape(Acal_test(:,:,i),1,n_tm*n_sm);
    end
    
    % Get classification performance on training and test sets
    [cc_sbt_train,cc_sbt_test] = ldacc(predictors_train,groups_train,...
        predictors_test,groups_test);

end


%% Troubleshooting / Plotting

% Compare noisy data with pristine data & factorized representation for
% first trial of the training set
offset = 0;
figure
for ii = 1:4
    %Display the noisey pattern that was factorized
    NoisyPattern = X_train(:,offset+(1:nBins)); 
    
    subplot(4,3,(ii-1)*3+1)
    imagesc(NoisyPattern)
    ylabel('Neuron ID')
    title(sprintf('Pattern %u with 50%% noise',ii))
    colorbar
    caxis([8,18])
    if ii == 4, xlabel('Time (ms)');end
    
    %Display the data without noise
    Pattern_NoNoise = data(1).counts{ii}{ind_train(1)};
    
    subplot(4,3,(ii-1)*3+2)
    imagesc(Pattern_NoNoise)
    title(sprintf('Pattern %u with no noise',ii))
    colorbar
    caxis([8,18])
    if ii == 4, xlabel('Time (ms)');end
    
    %Get the activation coefficients for this particular sample
    P = Wi*Acal_train(:,:,(ii-1)*10+1)*Wb;
    Pattern_NMF = P';
    
    subplot(4,3,(ii-1)*3+3)
    imagesc(Pattern_NMF)
    title(sprintf('Pattern %u factorized Representation',ii))
    colorbar
    caxis([8,18])
    if ii == 4, xlabel('Time (ms)');end
    
    Err = norm(NoisyPattern-Pattern_NMF,'fro')^2;
    
    %shift offset to get the first trial of the next pattern in X_train/X_test
    offset = offset + 300; 
end

% Compare noisy data with pristine data & factorized representation for
% first trial of the test set
offset = 0;
figure
for ii = 1:4
    %Display the noisey pattern that was factorized
    NoisyPattern = X_test(:,offset+(1:nBins)); 
    
    subplot(4,3,(ii-1)*3+1)
    imagesc(NoisyPattern)
    ylabel('Neuron ID')
    title(sprintf('Pattern %u with 50%% noise',ii))
    colorbar
    caxis([8,18])
    if ii == 4, xlabel('Time (ms)');end
    
    %Display the data without noise
    Pattern_NoNoise = data(1).counts{ii}{ind_test(1)};
    
    subplot(4,3,(ii-1)*3+2)
    imagesc(Pattern_NoNoise)
    title(sprintf('Pattern %u with no noise',ii))
    colorbar
    caxis([8,18])
    if ii == 4, xlabel('Time (ms)');end
    
    %Get the activation coefficients for this particular sample
    P = Wi*Acal_test(:,:,(ii-1)*10+1)*Wb;
    Pattern_NMF = P';
    
    subplot(4,3,(ii-1)*3+3)
    imagesc(Pattern_NMF)
    title(sprintf('Pattern %u factorized Representation',ii))
    colorbar
    caxis([8,18])
    if ii == 4, xlabel('Time (ms)');end
    
    Err = norm(NoisyPattern-Pattern_NMF,'fro')^2;
    
    %shift offset to get the first trial of the next pattern in X_train/X_test
    offset = offset + 300; 
end
% for ii = 1:4
%     subplot(2,2,ii)
%     imagesc(data(4).counts{ii,1}{1,1})
%     xlabel('Time');ylabel('Neuron ID')
%     title(sprintf('Pattern %u with 50 %% noise',ii))
%     colorbar
%     caxis([8,18])
% end

