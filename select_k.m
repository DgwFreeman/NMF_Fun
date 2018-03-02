function [DCmax, minSQE, n_m] = select_k(X_train,groups_train,X_test,groups_test,iN,iC)
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
noise = 0:0.1:1;
noise = [noise, [1.5, 2, 2.5, 3]];
fCoding = 0:0.06:0.84;

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

dtl = zeros(length(k_range),1);
for iK = 1:length(k_range)
    n_m = k_range(iK);
    % Decompose training set
    [W_train,H_train,~] = nmf(X_train,iK);

    % Obtain test set activation coefficients for given modules
    [~,H_test,~] = nmf(X_test,iK,W_train);
    
    %Save NMF results for later similarity analysis
    d = clock;
    datastr = sprintf('./NMFxvalN%uC%u_%u%.2u%.2u%.2u%.2u.mat',int16(noise(iN)*100),int16(fCoding(iC)*100),d(1:5));
    save(datastr,'W_train','H_Train','H_Test');
    
    % Process activation coefficients for classification
    predictors_train = H_train';
    predictors_test = H_test';
    [cc_train,cc_test] = ldacc(predictors_train,groups_train,predictors_test,groups_test);
    ctr(iK) = cc_train;
    cte(iK) = cc_test;
    
    %Calculate Squared Error
    sqerr_tr(iK) = norm(X_train - W_train*H_train,'fro')^2/norm(X_train,'fro')^2;
    sqerr_te(iK) = norm(X_test - W_train*H_test,'fro')^2/norm(X_test,'fro')^2;
    
    %% Determine if we've found the Squared Error 'kink', indicating the
    %point of diminishing returns for adding more components
    if iK > 2
        SQE_Slope = (sqerr_te(iK) - sqerr_te(1))/(iK - 1);
        b1 = sqerr_te(iK) - SQE_Slope*iK;
        xK =1:iK;
%         plot(xK,sqerr_te,'-ok'),hold on
%         plot(xK,SQE_Slope*xK + b1,'-r'),hold on
        
        ii = iK - 1;
        m2 = (-1/SQE_Slope);
        b2 = sqerr_te(ii) - m2*ii;
        
        %Point of intersection
        xx = (b2 - b1)/(SQE_Slope - m2);
        yy = SQE_Slope*xx + b1;
        
%         plot([ii,xx],[sqerr_te(ii),yy],'--b'),hold on
        dtl(ii,1) = sqrt((yy - sqerr_te(ii))^2 + (xx - ii)^2);

    end
    % Determine if we've found the 'Elbow' of the reconstruction error yet
    [currMax, currMaxIndex] = max(dtl);
    if prevMaxIndex == currMaxIndex, indy = indy + 1; end
    
    % Break the loop if we've found the elbow
    if indy > 2, break; end
    
    prevMax = currMax;
    prevMaxIndex = currMaxIndex;
            
end
minSQE = sqerr_te(currMaxIndex);
iK = currMaxIndex;
[DCmax,~] = max(cte(:));
n_m = k_range(iK);

end