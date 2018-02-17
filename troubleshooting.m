%% Troubleshooting / Plotting
% Compare noisy data with pristine data & factorized representation for
% first trial of the training set

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