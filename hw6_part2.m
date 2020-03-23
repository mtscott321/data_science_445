%% Parameters
D       = 100;
n       = 50;
discs = int16(sqrt(n));


%%
%{
dvals = [];
meanvals = [];
medianvals = [];
stdevvals = [];
for j =  20:10:100
    dvals = [dvals, j];
    dq = dist_data(j, n);
    meanvals = [meanvals, mean(dq)];
    medianvals = [medianvals, median(dq)];
    stdevvals = [stdevvals, std(dq)];
end
figure; subplot(1, 3, 1); scatter(dvals, meanvals); title("Means with varying Dimension");
subplot(1, 3, 2); scatter(dvals, medianvals); title("Medians with varying Dimension");
subplot(1, 3, 3); scatter(dvals, stdevvals); title("STD with varying Dimension");
%} 
%%
dims = [];
expecxs = [];
expecxys = [];
for k = 2:5:100
    sumx = 0; sumy = 0;
    for q = 1:1:5
        [ex, ey] = expected_sphere(k);
        sumx = sumx + ex; sumy = sumy + ey;
    end
    avgx = sumx/5; avgy = sumy/5;
    expecxs = [expecxs, avgx];
    expecxys = [expecxys, avgy];
    dims = [dims, k];
end
figure;
subplot(1, 2, 1); scatter(dims, expecxs); title("Expected norm(X)^2 (sphere) with varying dimension");
subplot(1, 2, 2); scatter(dims, expecxys); title("Expected norm(X-Y)^2 (sphere)with varying dimension");

    


%%
%Computing the expected value of norm(X)^2
function [expecx, expecxy] = expected(dim)
    num = 20;
    Id = eye(dim);
    nId = (1/dim) * eye(dim);
    mu = zeros(num, dim);
    X = mvnrnd(mu, Id, num);
    Y = mvnrnd(mu, Id, num);
    sumx = 0;
    sumxy = 0;
    for i  = 1:num
        sumx = sumx + (norm(X(i, :)))^2;
        sumxy = sumxy + (norm(X(i, :) - Y(i, :)))^2;
    end
    expecx = sumx/num;
    expecxy = sumxy/num;
end 

%% this one uses 1/d*Id
function [expecx, expecxy] = expected_norm(dim)
    num = 20;
    
    nId = (1/dim) * eye(dim);
    mu = zeros(num, dim);
    X = mvnrnd(mu, nId, num);
    Y = mvnrnd(mu, nId, num);
    sumx = 0;
    sumxy = 0;
    for i  = 1:num
        sumx = sumx + (norm(X(i, :)))^2;
        sumxy = sumxy + (norm(X(i, :) - Y(i, :)))^2;
    end
    expecx = sumx/num;
    expecxy = sumxy/num;
end 


%%
function distsQ = dist_data(num, dim)
    In = eye(dim);
    mu = zeros(num, dim);
    X = mvnrnd(mu, In, num);
    distsQ = pdist(X');
    fprintf('\nMin=%f\nMean=%f\nMedian=%f\nStdDev=%f\n',min(distsQ), mean(distsQ), median(distsQ), std(distsQ));
end

%%
function [expecx, expecxy] = expected_sphere(dim)
    num = 20;
    
    Id = eye(dim);
    mu = zeros(num, dim);
    X = mvnrnd(mu, Id, num);
    
    X = bsxfun(@rdivide,X,colnorms(X));
    
    
    
    Y = mvnrnd(mu, Id, num);
    X = bsxfun(@rdivide,X,colnorms(X));
    sumx = 0;
    sumxy = 0;
    for i  = 1:num
        sumx = sumx + (norm(X(i, :)))^2;
        sumxy = sumxy + (norm(X(i, :) - Y(i, :)))^2;
    end
    expecx = sumx/num;
    expecxy = sumxy/num;
end 

%%
function s = colnorms( X,p )

if nargin<2, p=2; end

if p<inf
    s = sum(abs(X).^p,1).^(1/p);
else
    s = max(abs(X),[],1);
end
end

