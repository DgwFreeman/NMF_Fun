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
nNeurons = 100;
nTrials = 20; 
nBins = 30; 
nPatterns = 4; % patterns of firing activity to be inferred
nStimuli=4; % stimulus conditions

%Create matrix for synthetic data
counts = cell(nPatterns,1);
trials = cell(nTrials,1);

%Gaussian noise to add to each pattern as a % of the signal
noise = [0, 0.1, 0.25, 0.5, 1];     
nSessions=numel(noise); % one session for each value of noise-> each session is one dataset 'counts'
data=repmat(struct('counts',counts),1,nSessions); % contains all datasets

%
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
    for iS = 1:nStimuli
        trials = cell(nTrials,1);
        %Base Firing Rate Pattern
        RatePattern = Patterns{iS}; % one pattern for each stimulus

        %For each neuron and time step, add gaussian noise
        for iTrial = 1:nTrials
            CountMatrix=repmat(RatePattern,1,nBins)+noise(iSess)*sigma_rate*randn(nNeurons,nBins);
            trials{iTrial,1} = CountMatrix;
        end
       counts{iS,1} = trials;
    end
    data(iSess).counts=counts;
end

figure(1); clf;
iS=1; iTrial=1;
for iSess=1:nSessions
    subplot(2,3,iSess)
    imagesc(data(iSess).counts{iS}{iTrial})
    xlabel('Time');ylabel('Neuron ID')
    title(sprintf('Stimulus%d,Trial%d[%0.03g noise]',iS,iTrial,noise(iSess)));
    colorbar
end
