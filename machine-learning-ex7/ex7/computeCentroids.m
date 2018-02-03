function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
K=size(idx,1);
for k = 1:K
    centroidsk = X(idx==k,:);
    assignedToCentroid = size(centroidsk,1);
    if(assignedToCentroid)
        avgx = sum(centroidsk)./assignedToCentroid;
        centroids(k,:)=avgx;
    end
end

% =============================================================

end
% =========================================================================
%!test
%! X = [   
%!   1.84208   4.60757;
%!   5.65858   4.79996;
%!   6.35258   3.29085
%!   ];
%! idx=[2;2;2];
%! K = 3;
%! centroids = computeCentroids(X, idx, K);
%! expans= [0 0;4.6177   4.2328;0 0]
%! assert(centroids, expans, .0001)

