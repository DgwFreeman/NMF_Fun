    %% Apply space-by-time NMF to factorize data    
    % Find optimal numbers of temporal and spatial modules
%     [n_tm,n_sm] = select_n_tm_n_sm(X_train,groups_train,n_e_train,X_test,groups_test,n_e_test);
%     
%     % Obtain temporal and spatial modules from training set
%     [Acal_train,Wi,Wb] = sbtnmf(X_train,n_tm,n_sm,n_e_train);
%     
%     % Obtain activation coefficients from test set for given modules
%     Acal_test = sbtnmf(X_test,n_tm,n_sm,n_e_test,Wi,Wb);
%     
%     % Process activation coefficients for classification
%     predictors_train = zeros(n_e_train,n_tm*n_sm);
%     for i = 1:n_e_train
%         predictors_train(i,:) = reshape(Acal_train(:,:,i),1,n_tm*n_sm);
%     end
%     predictors_test = zeros(n_e_test,n_tm*n_sm);
%     for i = 1:n_e_test
%         predictors_test(i,:) = reshape(Acal_test(:,:,i),1,n_tm*n_sm);
%     end
%     
%     % Get classification performance on training and test sets
%     [cc_sbt_train,cc_sbt_test] = ldacc(predictors_train,groups_train,predictors_test,groups_test);
%     
%     ctr(iSess) = cc_sbt_train;
%     cte(iSess) = cc_sbt_test;

% Compare noisy data with pristine data & factorized representation for
% first trial of the test set
% offset = 0;
% figure
% for ii = 1:4
%     %Display the noisey pattern that was factorized
%     NoisyPattern = X_test(:,offset+(1:nBins)); 
%     
%     subplot(4,3,(ii-1)*3+1)
%     imagesc(NoisyPattern)
%     ylabel('Neuron ID')
%     title(sprintf('Pattern %u with 50%% noise',ii))
%     colorbar
%     caxis([8,18])
%     if ii == 4, xlabel('Time (ms)');end
%     
%     %Display the data without noise
%     Pattern_NoNoise = data(1).counts{ii}{ind_test(1)};
%     
%     subplot(4,3,(ii-1)*3+2)
%     imagesc(Pattern_NoNoise)
%     title(sprintf('Pattern %u with no noise',ii))
%     colorbar
%     caxis([8,18])
%     if ii == 4, xlabel('Time (ms)');end
%     
%     %Get the activation coefficients for this particular sample
%     P = Wi*Acal_test(:,:,(ii-1)*10+1)*Wb;
%     Pattern_NMF = P';
%     
%     subplot(4,3,(ii-1)*3+3)
%     imagesc(Pattern_NMF)
%     title(sprintf('Pattern %u factorized Representation',ii))
%     colorbar
%     caxis([8,18])
%     if ii == 4, xlabel('Time (ms)');end
%     
%     Err = norm(NoisyPattern-Pattern_NMF,'fro')^2;
%     
%     %shift offset to get the first trial of the next pattern in X_train/X_test
%     offset = offset + 300; 
% end

for iSess = 1:nSessions
    fprintf('Noise Level: %u%%\n',noise(iSess)*100);
    X_test = TestCell{iSess};
    
    figure
    offset = 0;
    for iP = 1:nStimuli
        %Display one trial for each noisey pattern that was factorized
        NoisyPattern = X_test(:,offset+(1:nBins));
        
        subplot(4,3,(iP-1)*3+1)
        imagesc(NoisyPattern)
        ylabel('Neuron ID')
        title(sprintf('Pattern %u with %u%% noise',iP,noise(iSess)*100))
        colorbar
        caxis([8,18])
        if iP == 4, xlabel('Time (ms)');end
        
        %Display the data without noise
        Pattern_NoNoise = data(1).counts{iP}{ind_test(1)};
        
        subplot(4,3,(iP-1)*3+2)
        imagesc(Pattern_NoNoise)
        title(sprintf('Pattern %u with no noise',iP))
        colorbar
        caxis([8,18])
        if iP == 4, xlabel('Time (ms)');end
        
        %Get the activation coefficients for this particular sample
        %     P = Wi*Acal_train(:,:,(ii-1)*10+1)*Wb;
        P = SpatialModules{iSess}*TestCoeff{iSess};
        Pattern_NMF = P(:,offset+(1:nBins));
        
        subplot(4,3,(iP-1)*3+3)
        imagesc(Pattern_NMF)
        title(sprintf('Pattern %u factorized Representation',iP))
        colorbar
        caxis([8,18])
        if iP == 4, xlabel('Time (ms)');end
        
        Err = norm(NoisyPattern-Pattern_NMF,'fro')^2;
        
        %shift offset to get the first trial of the next pattern in X_train/X_test
        offset = offset + 300;
    end
end