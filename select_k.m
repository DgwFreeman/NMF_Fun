function n_m = select_k(X_train,groups_train,n_e_train,X_test,groups_test,n_e_test)
% Selects the optimal number of spatiotemporal modules.
% Input arguments:
%  X_train      - Training input matrix (size #cells
%                 x #time_points_per_trial * #training_set_trials)
%                 composed of horizontally concatenated population responses
%  groups_train - Class labels of each trial (e.g. stimulus identity);
%                 (vector of length #training_set_trials)
%  n_e_train    - Number of training set trials
%  X_test       - Test input matrix (size #cells
%                 x #time_points_per_trial * #test_set_trials)
%                 composed of horizontally concatenated population responses
%  groups_test  - Class labels of test set trials (e.g. stimulus identity);
%                 (vector of length #test_set_trials)
%  n_e_test     - Number of test set trials
% Output:
%  k            - Optimal number of spatial modules

Kmax = size(X_train,2)/2; % Maximum number of modules

%% Find optimal number of modules
k_range = 1:round(sqrt(Kmax));
% Percent correct classified on the training set as a function of #modules
ctr = zeros(length(k_range),1);
% Percent correct classified on the test set as a function of #modules
cte = zeros(length(k_range),1);
tStart = tic;

prevMax = 0;
prevMaxIndex = 0;
indy = 0;
for iK = 1:length(k_range)
    n_m = k_range(iK);
    % Decompose training set
    [W_train,H_train,~] = nmf(X_train,iK);

    % Obtain test set activation coefficients for given modules
    [~,H_Test,~] = nmf(X_test,iK,W_train);
    
    % Process activation coefficients for classification
    predictors_train = H_train';
    predictors_test = H_Test';
    [cc_train,cc_test] = ldacc(predictors_train,groups_train,predictors_test,groups_test);
    ctr(iK) = cc_train;
    cte(iK) = cc_test;

    % Determine if we've found a local max yet
    [currMax, currMaxIndex] = max(cte);
    if prevMaxIndex == currMaxIndex, indy = indy + 1; end
    % If we've had 3 consecutive iterations of i_s that resulted in no
    % improvement of classification performance, break the loop
    if indy > 2
        tElapsed = toc(tStart);
%         fprintf('Optimal # of Modules: %u, Time Elapsed: %3.3f\n', prevMaxIndex,tElapsed);
        break;
    end
    
    prevMax = currMax;
    prevMaxIndex = currMaxIndex;
        
end

[~,iK] = max(cte(:));
n_m = k_range(iK);

end