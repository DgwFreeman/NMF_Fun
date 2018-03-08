%% NMF Analysis Script for Angie's E-Phys Spike Train data for DOI 
load('C:\Users\Freeman\Documents\GitHub\NMF_Fun\drift_ephys_080715.mat')

%% Parameters
%Bin size in sec
bin_size = 0.01;
%Length of a trial in sec
len_trial = 2.5;
%# of Bins based on bin size
nBins = ceil(len_trial/bin_size);
%# of trials Pre & Post DOI
nTrialsPRE = size(drift_spikeT{1,1},2); 
nTrialsPOST = size(drift_spikeT{1,2},2); 
%Max # of trials per stimulus
nTrialsPerStim = 5;
%# of recorded Neurons
nNeurons = size(drift_spikeT{1,1},1);

%% Create Stimuli ID Matrix for sorting Data
nSF = 6;            %# of different Spatial Frequencies presented
nOR = 12;           %# of different Orientations presented
nStimuli = nOR*nSF; %# of unique stimuli presentations
%Add one more row for full field flicker stimulus presentation
StimulusID = NaN(nStimuli+1,2);
index = 1;
for i = 1:nSF
    for j = 1:nOR
        StimulusID(index,1) = i;
        StimulusID(index,2) = j;
        index = index + 1;
    end
end

%% Sort the data by stimulus
disp('Transforming spike trains to count matrices...');
counts = cell(nStimuli,2);

for iPrePost = 1:2
    for iStim = 1:nStimuli
        %Stimulus IDs
        iSF = StimulusID(iStim,1);
        iOR = StimulusID(iStim,2);
        
        %Pre & Post had a different amount of total trials
        if iPrePost == 1
            nTrials = nTrialsPRE;
        else
            nTrials = nTrialsPOST;
        end
        %Which trials showed spatial frequency iSF & orientation iOR
        if isnan(iSF)
            posSF = isnan(drift_sf_trial{1,iPrePost}(1:nTrials));
            posOR = isnan(drift_orient_trial{1,iPrePost}(1:nTrials));
        else
            posSF = drift_sf_trial{1,iPrePost}(1:nTrials) == iSF;
            posOR = drift_orient_trial{1,iPrePost}(1:nTrials) == iOR;
        end
        TrialIndices = find(posSF & posOR);
        
        %Create Trial cell array to contain trials of the same stimulus
        Trials = cell(length(TrialIndices),1);
        
        %Loop through each trial index and create a count matrix that is of
        %size (#ofNeurons by #ofTimeBins)
        SpikeTrain = drift_spikeT{1,iPrePost};
        for iTrial = 1:length(TrialIndices)
            tIndex = TrialIndices(iTrial);
            Trials{iTrial} = zeros(nNeurons,nBins);
            
            %Loop through each neuron and count the # of spikes in each timebin
            for iN = 1:nNeurons
                for iT = 1:nBins
                    Trials{iTrial}(iN,iT) = sum((SpikeTrain{iN,tIndex} > (iT-1)*bin_size) & (SpikeTrain{iN,tIndex} <= iT*bin_size));
                end
            end  
        end
        
        %Save the trial cell array for this stimulus presentation in counts
        counts{iStim,iPrePost} = Trials;
    end
end

%% Randomly separate data into training set and test set
%For Unsupervised Case, perform leave-2-out cross-validation procedure
%since there are only 5 trials maximum per stimulus
%Create nXVAL different combinations of the data split
nXVAL = nchoosek(nTrialsPerStim,2);
ind_train = nchoosek(1:1:nTrialsPerStim,3);
ind_test = flipud(nchoosek(1:1:nTrialsPerStim,2));

% Total number of training samples
n_e_train = (nStimuli)*size(ind_train,2);
% Total number of test samples
n_e_test = nStimuli*size(ind_test,2);

%% Model Selection
% Cross Validation to Select the Optimal Number of components
% Build overall training and test matrices
X_train = zeros(nNeurons,n_e_train*nBins);
X_test = zeros(nNeurons,n_e_test*nBins);
groups_train = zeros(n_e_train*nBins,1);
groups_test = zeros(n_e_test*nBins,1);

DCperf = zeros(nXVAL,1);
K_cv = zeros(nXVAL,1);
minSQE = zeros(nXVAL,1);
xvNMF = cell(nXVAL,3);

%Loop over x-validation runs
for iXV = 10:nXVAL
    offset_train = 0;
    offset_test = 0;
    
    %Loop over the different stiumuli & trials to create matrices
    for iStim = 1:nStimuli
        %Unsupervised x-validation, separate the training/test
        for iTrial = 1:size(ind_train,2)
            %Some Stimuli only have 4 trials & based on the trial indices
            %selected for x-validation, there may not be data from this
            %stimulus to concatenate onto the training data-set
            if ind_train(iXV,iTrial) <= length(counts{iStim,1})
                X_train(:,offset_train+(1:nBins)) = counts{iStim,1}{ind_train(iXV,iTrial)};
                groups_train(offset_train+(1:nBins),1) = iStim;
                offset_train = offset_train + nBins;
            end
        end
        for iTrial = 1:size(ind_test,2)
            %Some Stimuli only have 4 trials...
            if ind_test(iXV,iTrial) <= length(counts{iStim,2})
                X_test(:,offset_test+(1:nBins)) = counts{iStim,2}{ind_test(iXV,iTrial)};
                groups_test(offset_test+(1:nBins),1) = iStim;
                offset_test = offset_test + nBins;
            end
        end
    end
    
    %Take only the data we concatenated
    X_train = X_train(:,1:offset_train);
    X_test = X_train(:,1:offset_test);
    
    %Select the Optimal Number of components, k, using the unsupervised SQE
    %or the supervised decoding performance
    [DCperf(iXV),minSQE(iXV), K_cv(iXV), xvNMF{iXV,1}] = select_k(X_train,groups_train,X_test,groups_test,Opt);
    
    %Save matricces used for xval for later similarity analysis
    xvNMF{iXV,2} = X_train;
    xvNMF{iXV,3} = X_test;
end
