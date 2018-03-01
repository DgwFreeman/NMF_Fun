%% Troubleshooting / Plotting
% Compare noisy data with pristine data & factorized representation for
% first trial of the training set

clear all
cd C:\Users\Freeman\Documents\GitHub\NMF_Fun
NoInit = load('./Results/NoKmeansInit/NMFResults_201802161952.mat');
KmInit = load('./Results/KmeansInit/NMFResults_201802171538.mat');
load('./Results/KmeansInit/ExampleData_201802161635.mat','data','nPatterns',...
    'nCoding','nNoise','nStimuli','nTrials','nSessions','n_e_test',...
    'n_e_train','noise','fCoding','sigma_rate','Patterns');


%% Compare decoding performance for Kmeans and random seed Initializations 
figure
subplot(1,2,1)
imagesc(KmInit.dcte)
title('Decoding Performance for 4 Patterns using Kmeans Initialization');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([1,100])
colormap jet

subplot(1,2,2)
imagesc(NoInit.dcte)
title('Decoding Performance for 4 Patterns using Random Initialization');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([1,100])
colormap jet

%Just plot the difference
figure
imagesc(KmInit.dcte-NoInit.dcte)
title('Difference in Decoding Performance using Kmeans vs Random Seed Initializations');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([-5,5])
colormap jet

%% Compare the correlation between features of different runs
for iN = 1:nNoise
    for iC = 1:nCoding
        tmp = KmInit.mean_fCorr{iN,iC};
        tmp = triu(tmp,1);
        Km_fCorr = tmp(tmp ~= 0);
        
        %Calculate the mean of the mean correlation between features of
        %different runs
        Km_m_fCorr(iN,iC) = mean(Km_fCorr);
        Km_s_fCorr(iN,iC) = std(Km_fCorr);
        
    end
end

for iN = 1:nNoise
    for iC = 1:nCoding
        tmp = NoInit.mean_fCorr{iN,iC};
        tmp = triu(tmp,1);
        m_fCorr = tmp(tmp ~= 0);
        
        %Calculate the mean of the mean correlation between features of
        %different runs
        No_m_fCorr(iN,iC) = mean(m_fCorr);
        No_s_fCorr(iN,iC) = std(m_fCorr);
        
    end
end

figure
subplot(2,2,1)
imagesc(Km_m_fCorr)
title('Mean Correlation Coefficient using Kmeans Initialization');
ylabel('% Additive Noise')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet

subplot(2,2,2)
imagesc(No_m_fCorr)
title('Mean Correlation Coefficient using Random Initialization');
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet

subplot(2,2,3)
imagesc(Km_s_fCorr)
title('Std Correlation Coefficient using Kmeans Initialization');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet

subplot(2,2,4)
imagesc(No_s_fCorr)
title('Std Correlation Coefficient using Random Initialization');
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet
suptitle('Mean Correlation Coefficient between features of different runs');

%% Compare the correlation between coefficients of different runs
for iN = 1:nNoise
    for iC = 1:nCoding
        tmp = KmInit.mean_tcCorr{iN,iC};
        tmp = triu(tmp,1);
        Km_fCorr = tmp(tmp ~= 0);
        
        %Calculate the mean of the mean correlation between features of
        %different runs
        Km_m_fCorr(iN,iC) = mean(Km_fCorr);
        Km_s_fCorr(iN,iC) = std(Km_fCorr);
        
    end
end

for iN = 1:nNoise
    for iC = 1:nCoding
        tmp = NoInit.mean_tcCorr{iN,iC};
        tmp = triu(tmp,1);
        m_fCorr = tmp(tmp ~= 0);
        
        %Calculate the mean of the mean correlation between features of
        %different runs
        No_m_fCorr(iN,iC) = mean(m_fCorr);
        No_s_fCorr(iN,iC) = std(m_fCorr);
        
    end
end

figure
subplot(2,2,1)
imagesc(Km_m_fCorr)
title('Mean Correlation Coefficient using Kmeans Initialization');
ylabel('% Additive Noise')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet

subplot(2,2,2)
imagesc(No_m_fCorr)
title('Mean Correlation Coefficient using Random Initialization');
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet

subplot(2,2,3)
imagesc(Km_s_fCorr)
title('Std Correlation Coefficient using Kmeans Initialization');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet

subplot(2,2,4)
imagesc(No_s_fCorr)
title('Std Correlation Coefficient using Random Initialization');
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([0,1])
colormap jet
suptitle('Mean Correlation Coefficient between Activation Coefficients of different runs');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cc = hsv(nNoise);
for iN = 1:nNoise
    fprintf('Plotting %u%% noise level...\n',int16(noise(iN)*100));
    
    for iC = 1:nCoding
        fprintf('\t %u%% non-coding patterns introduced...\n',int16(fCoding(iC)*100));
        
        %% Build overall data matrices
        X_train = zeros(nNeurons,n_e_train*nBins);
        X_test = zeros(nNeurons,n_e_test*nBins);
        offset = 0;
        for iStim = 1:nStimuli
            for iTrial = 1:length(ind_train)
                X_train(:,offset+(1:nBins)) = data(iN,iC).counts{iStim}{ind_train(iTrial)};
                X_test(:,offset+(1:nBins)) = data(iN,iC).counts{iStim}{ind_test(iTrial)};
                
                offset = offset + nBins;
            end
        end
        
        %Normalized Test Matrix and Factorized Matrix representation
        NoiseyPattern = X_test;
        P = SpatialModules{iN,iC}*TestCoeff{iN,iC};
        Pattern_NMF = P;
        for iT = 1:(nBins*nTrials*nPatterns/2)
            NoiseyPattern(:,iT) = NoiseyPattern(:,iT)/norm(NoiseyPattern(:,iT));
            Pattern_NMF(:,iT) = Pattern_NMF(:,iT)/norm(Pattern_NMF(:,iT));
        end
        
        ss = sprintf('Noise Level: %u%% -- Non-Coding Percentage: %u%%',int16(noise(iN)*100),int16(fCoding(iC)*100));
        figure
        suptitle(ss);
        
        subplot(1,3,1)
        imagesc(NoiseyPattern)
        colorbar
        caxis([0,1])
        title('Firing Rate Data')
        xlabel('Time(ms)')
        ylabel('Neuron ID')
        
        subplot(1,3,2)
        imagesc(Pattern_NMF)
        colorbar
        caxis([0,1])
        title('Factorized Representation')
        xlabel('Time(ms)')
        ylabel('Neuron ID')
        
        
        
    end
end

for ii = 1:20
    subplot(4,5,ii)
    imagesc(data(ii).counts{1,1}{1,1})
    xlabel('Time');ylabel('Neuron ID')
    title(sprintf('%2.2f %%',fCoding(ii)))
    colorbar
    caxis([8,18])
end

figure
plot(noise*100,dcte,'.k','MarkerSize',15)
xlabel('Percent Additive Noise')
ylabel('Decoding Performance')
title('Decoding Performance for 4 Patterns');

figure
imagesc(dcte)
title('Decoding Performance for 4 Patterns');
ylabel('% Additive Noise')
xlabel('% Non-Coding Patterns')
xticks(1:1:15)
yticks(1:1:15)
xticklabels(fCoding*100);
yticklabels(noise*100);
colorbar
caxis([1,100])
colormap jet


cc = hsv(nNoise);
tvec = 1:1:(nBins*nTrials*nPatterns/2);
for iN = 1:nNoise
    
    fprintf('Plotting %u%% noise level...\n',int16(noise(iN)*100));
    figure
    for iC = 1:nCoding
        
        Coeff = TestCoeff{iN,iC};
        for iM = 1:kFeat(iN,iC)
            plot(tvec,Coeff(iM,:),'-','Color',cc(iC,:)),hold on
        end
        
    end
end

testBasis = SpatialModules{1, 4};
for i = 1:4
    testPatterns(:,i) = Patterns{i, 1};
    tt = norm(testPatterns(:,i));
    testPatterns(:,i) = testPatterns(:,i)/tt;
    
    tt = norm(testBasis(:,i));
    testBasis(:,i) = testBasis(:,i)/tt;
    
end
figure;
subplot(2,1,1);
imagesc(testPatterns);
colorbar
caxis([0,1])

subplot(2,1,2);
imagesc(testBasis)
colorbar
caxis([0,1])

%% Show factorized matrices
figure
subplot(k,3,1:3:k*3)
imagesc(X_test)

subplot(k,3,2:3:k*3)
imagesc(W_train)

tvec = 1:1:length(H_Test);
for iP = 1:k
    subplot(k,3,iP*3)
    plot(tvec,H_Test(iP,:),'-','LineWidth',2)
end




ss = sprintf('Noise Level: %u%% -- Non-Coding Percentage: %u%% ',int16(noise(iN)*100),int16(fCoding(iC-2)*100));

figure
imagesc(mean_fCorr{iN,iC-2})
colorbar
caxis([0.7,1])
title('Mean Correlation of Features between NMF runs');
suptitle(ss);
xlabel('Run j'); ylabel('Run i');
            
figure
imagesc(mean_tcCorr{iN,iC-2})
colorbar
caxis([0.5,1])
title('Mean Correlation of Activation Coefficients between NMF runs');
suptitle(ss);
xlabel('Run j'); ylabel('Run i');

            
            subplot(1,2,2)
            imagesc(std_fCorr)
            colorbar
            caxis([0,1])
            title('Std Correlation of Features between NMF runs');
            xlabel('Run j'); ylabel('Run i');
            
            
            
            figure
            ss = sprintf('Noise Level: %u%% -- Non-Coding Percentage: %u%% ',int16(noise(iN)*100),int16(fCoding(iC)*100));
            suptitle(ss);
            
            subplot(2,1,1)
            imagesc(Caa);
            colormap jet
                        colorbar
            caxis([-1,1])
            title('Correlation of Features between NMF runs 1 & 1');
            xlabel('Feature j'); ylabel('Feature i');
            
            subplot(2,1,2)
            imagesc(Cab);
            colormap jet
                        colorbar
            caxis([-1,1])
            title('Correlation of Features between NMF runs 1 & 2');
            xlabel('Feature j'); ylabel('Feature i');
            
            
            histogram(histCorr,20)
xlabel('Correlation Coefficient')
title('Histogram of Correlation Coefficients of corresponding features of different runs')
      

figure
ss = sprintf('Noise Level: %u%% -- Non-Coding Percentage: %u%%',int16(noise(iN)*100),int16(fCoding(iC)*100));
suptitle(ss);
subplot(1,2,1)
imagesc(X_train)
colorbar
caxis([8,18])
title('Firing Rate Data')
xlabel('Time(ms)')
ylabel('Neuron ID')

subplot(1,2,2)
Pattern_NMF = W_train*H_train;
imagesc(Pattern_NMF)
colorbar
caxis([8,18])
title('Factorized Representation')
xlabel('Time(ms)')
ylabel('Neuron ID')


figure
plot(noise*100,dcte(:,1),'-ok')
ylabel('Decoding Performance')
xlabel('% Noise Added')
title('With K-Means Initialization')
grid on

figure
cc = hsv(nNoise);
lgnd_str = cell(nNoise,1);
for iC = 1:nCoding
    for iN = 1:nNoise
   
    tmp = mean_fCorr{iN,iC};
    tmp = triu(tmp,1);
    m_fCorr = tmp(tmp ~= 0);
    
    %Calculate the mean of the mean correlation between features of
    %different runs
    mm_fCorr(iN,1) = mean(m_fCorr);
    ss_fCorr(iN,1) = std(m_fCorr); 
    
    end
    
    plot(noise*100,mm_fCorr,'-.','MarkerSize',8,...
    'MarkerEdgeColor',cc(iC,:),'MarkerFaceColor',cc(iC,:)),hold on
%     errorbar(noise*100,mm_fCorr,ss_fCorr,'-s','MarkerSize',8,...
%     'MarkerEdgeColor',cc(iC,:),'MarkerFaceColor',cc(iC,:)),hold on
    ss = sprintf('Non-Coding Percentage: %u%%',int16(fCoding(iC)*100));
    lgnd_str{iC,1} = {ss};

end
figure
errorbar(noise*100,mm_fCorr,std_fCorr,'-s','MarkerSize',8,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')

xlabel('% Noise Added')
ylabel('Mean Correlation')
title('Mean of the Mean Correlation betwen features of different runs')


        %% Optional Plotting
        if plot_figs == 1
            figure
            ss = sprintf('Noise Level: %u%% -- Non-Coding Percentage: %u%% -- DC Performance: %2.1f%%',int16(noise(iN)*100),int16(fCoding(iC)*100),cc_test);
            suptitle(ss);
            offset = 0;
            for iP = 1:nStimuli
                %Display one trial for each noisey pattern that was factorized
                NoisyPattern = X_test(:,offset+(1:nBins));
                
                subplot(4,3,(iP-1)*3+1)
                imagesc(NoisyPattern)
                ylabel('Neuron ID')
                title(sprintf('Pattern %u with noise',iP))
                colorbar
                caxis([8,18])
                if iP == 4, xlabel('Time (ms)');end
                
                %Display the data without noise
                Pattern_NoNoise = data(1).counts{iP}{ind_test(1)};
                
                subplot(4,3,(iP-1)*3+2)
                imagesc(Pattern_NoNoise)
                title('With no noise')
                colorbar
                caxis([8,18])
                if iP == 4, xlabel('Time (ms)');end
                
                %Get the activation coefficients for this particular sample
                %     P = Wi*Acal_train(:,:,(ii-1)*10+1)*Wb;
                P = SpatialModules{iN,iC}*TestCoeff{iN,iC};
                Pattern_NMF = P(:,offset+(1:nBins));
                
                subplot(4,3,(iP-1)*3+3)
                imagesc(Pattern_NMF)
                title('Factorized Representation')
                colorbar
                caxis([8,18])
                if iP == 4, xlabel('Time (ms)');end
                
                Err = norm(NoisyPattern-Pattern_NMF,'fro')^2;
                
                %shift offset to get the first trial of the next pattern in X_train/X_test
                offset = offset + nBins*nTrials/2;
            end
        end
        
        
        
        
            dtl = NaN(iK - 2,1);
    if iK > 2
        SQE_Slope = (sqerr_te(iK) - sqerr_te(1))/(iK - 1);
        b1 = sqerr_te(1) - SQE_Slope;
        xK =1:iK;
        plot(xK,sqerr_te,'-ok'),hold on
        plot(xK,SQE_Slope*xK + b1,'-r'),hold on
        
        for ii = 2:iK-1
            b2 = sqerr_te(ii) + SQE_Slope*ii;
            
            %Point of intersection
            xx = (b2 - b1)/(2*SQE_Slope);
            yy = SQE_Slope*xx + b1;
            
            plot([ii,xx],[sqerr_te(ii),yy],'b'),hold on
            dtl(ii,1) = sqrt((yy - sqerr_te(ii))^2 + (xx - ii)^2);

        end
        [currMax, currMaxIndex] = max(dtl);
        dist_to_line{iK} = dtl;
    end
    
        %Determine if we've found the Squared Error 'Elbow'
    if iK > 2
        SQE_Slope = (sqerr_te(iK) - sqerr_te(iK-2))/2;
        b1 = sqerr_te(iK) - SQE_Slope*iK;
        xK =iK-2:iK;
        plot(1:iK,sqerr_te,'-ok'),hold on
        plot(xK,SQE_Slope*xK + b1,'-r'),hold on
        
        ii = iK - 1;
        b2 = sqerr_te(ii) + SQE_Slope*ii;
        
        %Point of intersection
        xx = (b2 - b1)/(2*SQE_Slope);
        yy = SQE_Slope*xx + b1;
        
        plot([ii,xx],[sqerr_te(ii),yy],'b'),hold on
        dtl(iK,1) = sqrt((yy - sqerr_te(ii))^2 + (xx - ii)^2);

    end
    
    
                %% Decompose training set with VSMF function
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