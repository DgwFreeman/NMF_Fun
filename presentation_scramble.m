

pTbins = randperm(size(X_test,2));

load('C:\Users\Freeman\Documents\GitHub\NMF_Fun\X_testnonoise.mat');
%For X_test with Noise
X = X_test(:,pTbins(1:100));  
figure; imagesc(X),hold on
xticks([]);yticks([]);
plot([25,25],[0,21],'-w','LineWidth',3),hold on
plot([50,50],[0,21],'-w','LineWidth',3),hold on
plot([75,75],[0,21],'-w','LineWidth',3),hold on
set(gca,'YTick',[1 10 20],'YTickLabel',{'1','10','20'},'XTick', [12 37 62 87],'XTickLabel',{'Trial 1','Trial 2','Trial 3','Trial 4'}, 'fontsize', 18);


figure; imagesc(X(:,1:50)),hold on
xticks([]);yticks([]);
plot([25,25],[0,21],'-w','LineWidth',3),hold on
set(gca,'YTick',[1 10 20],'YTickLabel',{'1','10','20'},'XTick', [12 37],'XTickLabel',{'Trial 1','Trial 2'}, 'fontsize', 18);

figure; imagesc(X(:,51:100)),hold on
xticks([]);yticks([]);
plot([25,25],[0,21],'-w','LineWidth',3),hold on
set(gca,'YTick',[1 10 20],'YTickLabel',{'1','10','20'},'XTick', [12 37],'XTickLabel',{'Trial 3','Trial 4'}, 'fontsize', 18);


[W,H,err] = nmf(X,4);
figure; imagesc(H)
xticks([]);yticks([]);
H_Norm = zeros(size(H));
for iBin = 1:size(H,2)
   H_Norm(:,iBin) = H(:,iBin)./norm(H(:,iBin));
end
figure
for i = 1:4
    subplot(4,1,i);
    plot(1:40,H_Norm(i,:),'.-k','LineWidth',2),hold on
    xticks([])
    ylim([0,1])
    set(gca,'YTick',[0 0.5 1],'fontsize', 18);
end
set(gca,'YTick',[0 0.5 1],'XTick', [1 10 20 30 40],'fontsize', 18);

X_tilda = W*H;
SQE = norm(X - X_tilda,'fro')^2/norm(X,'fro')^2;
figure; imagesc(X_tilda)
set(gca,'YTick',[1 10 20],'YTickLabel',{'1','10','20'},'XTick', [1 10 20 30 40],'XTickLabel',{'1','10','20','30','40'}, 'fontsize', 18);

load('C:\Users\Freeman\Documents\GitHub\NMF_Fun\X_testnonoise.mat');
%% For X_test without Noise
X = X_test(:,pTbins(1:40));  
figure; imagesc(X)
xticks([]);yticks([]);
set(gca,'YTick',[1 10 20],'YTickLabel',{'1','10','20'},'XTick', [1 10 20 30 40],'XTickLabel',{'1','10','20','30','40'}, 'fontsize', 18);

[W,H,err] = nmf(X,4);
figure; imagesc(W)
xticks([]);yticks([]);
for iBin = 1:size(H,2)
   H(:,iBin) = H(:,iBin)./norm(H(:,iBin));
end
figure
for i = 1:4
    subplot(4,1,i);
    plot(1:40,H(i,:),'.-k','LineWidth',2),hold on
    xticks([])
    ylim([0,1])
    set(gca,'YTick',[0 0.5 1],'fontsize', 12);
end

set(gca,'YTick',[0 0.5 1],'XTick', [1 10 20 30 40],'fontsize', 12);



X_tilda = W*H;
figure; imagesc(X_tilda)
set(gca,'YTick',[1 10 20],'YTickLabel',{'1','10','20'},'XTick', [1 10 20 30 40],'XTickLabel',{'1','10','20','30','40'}, 'fontsize', 18);