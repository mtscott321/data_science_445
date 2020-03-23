%% Parameters
D       = 100;
n       = 1000;
discs = int16(sqrt(n));
%% Unit cube
%{
X = rand(D,n);
distsQ = pdist(X');
distsQnn = pdist2(X(:,1)',X(:,2:end)');
fprintf('\nMin=%f\nMean=%f\nMedian=%f\nStdDev=%f\n',min(distsQnn), mean(distsQnn), median(distsQnn), std(distsQnn));
figure; 
%each subplot graphs a different thing in the main figure
subplot(1,1,1);%hist((distsQ(:) - mean(distsQ))/std(distsQ),1000);
hold on
%here we're gonna plot the gaussian
hold off
f = (distsQ(:) - mean(distsQ))/std(distsQ);

fprintf('Normalized distsQ\nMin=%f\nMean=%f\nMedian=%f\nStdDev=%f\n',min(f), mean(f), median(f), std(f));

[distsQ_bins, edges] = histcounts((distsQ(:) - mean(distsQ))/std(distsQ), discs);
%norm = randn(size(distsQ));
%[norm_bins, edges] = histcounts(, discs);
bin_centers = (edges(2:end) - edges(1:end-1))/2;

%subplot(1, 2, 2); hist(norm, 1000);
%makedist('Normal','mu',0, 'sigma',1)
%subplot(1,2,2); hist(distsQnn,100);

sum = 0;

for i = 1:discs
    total = 0;
    for j = 1:10
        %disp(abs((norm_bins(i) - distsQ_bins(i))));
        total = total + abs((gauss(bin_centers(i)) - distsQ_bins(i)));
    end
    sum = sum + total/10
end
diff = sum/discs;
disp(diff);
%}

%%




x = [];
y = [];
for j = 2:25:1000
    diff = calc_diff(j, n, discs);
    x = [x, j];
    y = [y, diff];
end

testx = isvector(x);
figure; subplot(1, 1, 1); scatter(x, y);




%% Unit sphere
%{
X = randn(D,n);
X = bsxfun(@rdivide,X,colnorms(X));

distsS = pdist(X');
distsSnn = pdist2(X(:,1)',X(:,2:end)');
fprintf('\nMin=%f\nMean=%f\nMedian=%f\n',min(distsSnn), mean(distsSnn), median(distsSnn));
figure; subplot(1,2,1);hist(distsS(:),1000);subplot(1,2,2); hist(distsSnn,100);
%}
%% making the x values into spherical ones
function s = colnorms( X,p )

if nargin<2, p=2; end

if p<inf
    s = sum(abs(X).^p,1).^(1/p);
else
    s = max(abs(X),[],1);
end
end

%%
function diff = calc_diff(D, n, discs)
X = randn(D,n);
X = bsxfun(@rdivide,X,colnorms(X));
distsQ = pdist(X');
[distsQ_bins, edges] = histcounts((distsQ(:) - mean(distsQ))/std(distsQ), discs);
bin_centers = (edges(2:end) - edges(1:end-1))/2;
sum = 0;
for i = 1:discs
    total = 0;
    for j = 1:10
        %disp(abs((norm_bins(i) - distsQ_bins(i))));
        total = total + abs((gauss(bin_centers(i)) - distsQ_bins(i)));
    end
    sum = sum + total/10
end
diff = sum/discs;
end


function y = gauss(x)
y = 1/(2*pi)*exp(-(x).^2/(2));
end
