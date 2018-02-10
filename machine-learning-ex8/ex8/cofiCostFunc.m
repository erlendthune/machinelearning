function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

tmpJ = 0;
for i = 1:num_movies
    for j= 1:num_users
        if(R(i,j)==1)
            tmpJ += (Theta(j,:)*X(i,:)'-Y(i,j))^2;
        end
    end
end
J = tmpJ / 2;














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

%!test
%! Theta = [
%!   0.28544  -1.68427   0.26294;
%!   0.50501  -0.45465   0.31746;
%!  -0.43192  -0.47880   0.84671;
%!   0.72860  -0.27189   0.32684;
%!   ];
%! X = [   
%!   1.048686  -0.400232   1.194119;
%!   0.780851  -0.385626   0.521198;
%!   0.641509  -0.547854  -0.083796;
%!   0.453618  -0.800218   0.680481;
%!   0.937538   0.106090   0.361953;
%!   ];
%! Y = [
%!    5   4   0   0;
%!    3   0   0   0;
%!    4   0   0   0;
%!    3   0   0   0;
%!    3   0   0   0;
%!  ];
%! R = [
%!    1   1   0   0;
%!    1   0   0   0;
%!    1   0   0   0;
%!    1   0   0   0;
%!    1   0   0   0;
%! ];
%! num_users = 4; num_movies = 5; num_features = 3;
%! J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 0)
%! expans = 22.22;
%! assert(J, expans, .005)
