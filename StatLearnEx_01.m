%% Parameters
%D       = 2;
n       = 40;
K       = 10;
varmod  = 1/2;
Knn     = 10;
rounds  = 10;
means   = [];
l = 0.2;


%%
ds = [];
knn_err = [];
ls_err = [];
D = 0;
for i = 2:1:15
    D = i;
   
    ds = [ds, i];
    
    [X,Y,means]     = generateData( n,D,K, means,varmod, l );
    [Xtest,Ytest]   = generateData( n,D,K, means,varmod, l );
    [Yhatcont,Yhat_LS,cm_LS] = fitLS( X,Y, Xtest, Ytest );
    [Yhat_Knn,cm_Knn] = fitKnn( X,Y, Knn, Xtest, Ytest );
    err_LS = 1-trace(cm_LS)/sum(sum(cm_LS));
    err_Knn = 1-trace(cm_Knn)/sum(sum(cm_Knn));
    
    ls_err = [ls_err, err_LS];
    knn_err = [knn_err, err_Knn];
    
    
end

figure;
subplot(1,2,1); scatter(ds, ls_err); title("Least Square Error");
subplot(1, 2, 2); scatter(ds, knn_err); title("KNN error");


%%
for vm = 0.1:0.1:2
    
    ds = [ds, vm];
    
    [X,Y,means]     = generateData( n,D,K, means,vm, l );
    [Xtest,Ytest]   = generateData( n,D,K, means,vm, l );
    [Yhatcont,Yhat_LS,cm_LS] = fitLS( X,Y, Xtest, Ytest );
    [Yhat_Knn,cm_Knn] = fitKnn( X,Y, Knn, Xtest, Ytest );
    err_LS = 1-trace(cm_LS)/sum(sum(cm_LS));
    err_Knn = 1-trace(cm_Knn)/sum(sum(cm_Knn));
    
    ls_err = [ls_err, err_LS];
    knn_err = [knn_err, err_Knn];

end

figure; title("Varying varmod");
subplot(1,2,1); scatter(ds, ls_err); title("Least Square Error");
subplot(1, 2, 2); scatter(ds, knn_err); title("KNN error");

%%
min_vals = [];
min_err = 1000000000000000;
for lambda = 0.1:0.1:3
    for vm = 0.1:0.1:3
        [X,Y,means]     = generateData( n,D,K, means,vm, l );
        [Xtest,Ytest]   = generateData( n,D,K, means,vm, l );
        [Yhatcont,Yhat_LS,cm_LS] = fitLS( X,Y, Xtest, Ytest );
        [Yhat_Knn,cm_Knn] = fitKnn( X,Y, Knn, Xtest, Ytest );
        err_LS = 1-trace(cm_LS)/sum(sum(cm_LS));
        err_Knn = 1-trace(cm_Knn)/sum(sum(cm_Knn));
        
        if err_Knn < min_err
            min_err = err_Knn;
            min_vals = [lambda, vm];
        end
    end
end

        

%% Generate data
[X,Y,means] = generateData( n,D,K,means,varmod, l );

%% Classify using Least Squares
[Yhatcont,Yhat_LS,cm_LS] = fitLS( X,Y,X,Y );

cm_LS

figure;
subplot(1,2,1);scatter(X(1,:),X(2,:),20,Y,'filled');
subplot(1,2,2);scatter(X(1,:),X(2,:),20,Yhat_LS,'filled');
set(gcf,'Name',sprintf('LS; error = %.2f',1-trace(cm_LS)/sum(sum(cm_LS))));

%% Classifying using nearest neighbors
[Yhat_Knn, cm_Knn] = fitKnn( X,Y, Knn,X,Y );

cm_Knn

figure;
subplot(1,2,1);scatter(X(1,:),X(2,:),20,Y,'filled');
subplot(1,2,2);scatter(X(1,:),X(2,:),20,Yhat_Knn,'filled');
set(gcf,'Name',sprintf('kNN; error = %.2f',1-trace(cm_Knn)/sum(sum(cm_Knn))));

%% Do this for increasing n
ns = round(2.^(8:0.25:16));

clear err_LS err_Knn

for l = 1:length(ns)
    n = ns(l);
    for r = 1:rounds
        [X,Y,means]     = generateData( n,D,K, means,varmod, l );
        [Xtest,Ytest]   = generateData( n,D,K, means,varmod, l );
        [Yhatcont,Yhat_LS,cm_LS] = fitLS( X,Y, Xtest, Ytest );
        [Yhat_Knn,cm_Knn] = fitKnn( X,Y, Knn, Xtest, Ytest );
        err_LS(l,r) = 1-trace(cm_LS)/sum(sum(cm_LS));
        err_Knn(l,r) = 1-trace(cm_Knn)/sum(sum(cm_Knn));
    end
end

figure;plot(ns,mean(err_LS,2)); hold on;plot(ns,mean(err_Knn,2));axis tight
xlabel('n'); ylabel('err');


%% Do this for increasing Knn
KNNs = [1:10, 10:20:n];

clear err_Knn err_Knn_train

for l = 1:length(KNNs)
    for r = 1:rounds
        [X,Y,means]     = generateData( n,D,K, means,varmod, l);
        [Xtest,Ytest]   = generateData( n,D,K, means,varmod , l);
        [Yhat_Knn,cm_Knn] = fitKnn( X,Y, KNNs(l), X, Y );
        err_Knn_train(l,r) = 1-trace(cm_Knn)/sum(sum(cm_Knn));
        [Yhat_Knn,cm_Knn] = fitKnn( X,Y, KNNs(l), Xtest, Ytest );
        err_Knn(l,r) = 1-trace(cm_Knn)/sum(sum(cm_Knn));
    end
end

figure;plot(n./KNNs,mean(err_Knn,2));hold on; plot(n./KNNs,mean(err_Knn_train,2)); axis tight
xlabel('n/Knn'); ylabel('err');legend('test','train');

%% Data generator
function [X,Y,means] = generateData( n,D,K, means, varmod , l)

%% Generate data distribution
if isempty(means)
    for c = 1:2
        means{c}        = randn(D,K);
        means{c}(c,:)   = means{c}(c,:)+1;
    end
end

%% Generate data set
for c = 1:2
    Xc{c} = sample(n ,D, K, means{c}, varmod, l );
    Yc{c} = sign(c-1.5)*ones(1,n);
end

X = [Xc{:}];
Y = [Yc{:}];
end


%% sampling function
function X = sample( n, D, K, means, varmod, l)

X           = l^2 *varmod*randn(2,n);


modeidxs    = unidrnd(K,n,1);

for q = 1:n
    X(:,q) = X(:,q) + means(:,modeidxs(q));
end

end

%% LS
function [Yhatcont,Yhat_LS,cm] = fitLS( X,Y, Xtest, Ytest )

%% Classify using Least Squares
%alpha = pinv(X)'*Y';
alpha = X'\Y';
Yhatcont = (Xtest'*alpha)';

Yhat_LS(Yhatcont>0) = 1;
Yhat_LS(Yhatcont<=0) = -1;

cm = confusionmat(Ytest,Yhat_LS);

end

%% KNN
function [Yhat_Knn,cm] = fitKnn( X,Y,Knn,Xtest,Ytest )
%% Classifying using nearest neighbors
MdlKnn = fitcknn(X',Y,'NumNeighbors',Knn,'Standardize',1);
Yhat_Knn = MdlKnn.predict(Xtest');

cm = confusionmat(Ytest,Yhat_Knn);
end
