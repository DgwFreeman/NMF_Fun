function [DCmax, minSQE, n_m] = select_k(X_train,groups_train,X_test,groups_test,Opt)
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
% Output Arguments
%  Output struct
%
%  k            - Optimal number of spatial modules
%%
noise = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3];
fCoding = 0:0.1:0.9;

Kmax = round(size(X_train,2)/2); % Maximum number of modules

%% Find optimal number of modules
% Percent correct classified 
ctr = zeros(Kmax,1);
cte = zeros(Kmax,1);

%Indices to determine stopping criteria
prevMax = 0;
prevMaxIndex = 0;
indy = 0;
dtl = zeros(Kmax,1);

%Save 

%Save the test and training data for later use
d = clock;
datastr = sprintf('./NMFxvalN%uC%u_%u%.2u%.2u%.2u%.2u%.2u.mat',int16(noise(Opt.iN)*100),int16(fCoding(Opt.iC)*100),d(1:5),floor(d(6)));
save(datastr,'X_train','X_test');

for iK = 1:Kmax
    % Decompose training set
    [W_train,H_train,~] = nmf(X_train,iK);

    % Obtain test set activation coefficients for given modules
    [~,H_test,~] = nmf(X_test,iK,W_train);
    
    %Save NMF results for later similarity analysis
    d = clock;
    datastr = sprintf('./NMFxvalN%uC%uK%u_%u%.2u%.2u%.2u%.2u%.2u.mat',int16(noise(Opt.iN)*100),int16(fCoding(Opt.iC)*100),iK,d(1:5),floor(d(6)));
    save(datastr,'W_train','H_train','H_test');
    
    % Process activation coefficients for classification
    predictors_train = H_train';
    predictors_test = H_test';
    [cc_train,cc_test] = ldacc(predictors_train,groups_train,predictors_test,groups_test);
    ctr(iK) = cc_train;
    cte(iK) = cc_test;
    
    %Calculate Squared Error
    sqerr_tr(iK) = norm(X_train - W_train*H_train,'fro')^2/norm(X_train,'fro')^2;
    sqerr_te(iK) = norm(X_test - W_train*H_test,'fro')^2/norm(X_test,'fro')^2;
    
    %% Stopping Criteria 
    if Opt.SV == 1
        %For supervised x-validation
        %Determine if we've found a maximum decoding performance yet
        [currMax, currMaxIndex] = max(cte);
        %Is it getting better?
        if prevMaxIndex == currMaxIndex, indy = indy + 1; end
        
    else
        %% For Unsupervised x-validation
        %Determine if we've found the Squared Error 'kink', indicating the
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
    end

    % Break the loop if we've found the elbow
    if indy > 2, break; end
    
    prevMax = currMax;
    prevMaxIndex = currMaxIndex;
            
end

%% Outputs
%Output the minumum reconstruction error
minSQE = sqerr_te(currMaxIndex);
%Output the number of parameters selected
iK = currMaxIndex;
n_m = k_range(iK);
%Output the decoding performance for that k-value
[DCmax,~] = max(cte(:));

end