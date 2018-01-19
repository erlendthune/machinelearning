function [theta] = trainLinearReg(X, y, lambda)
%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
%regularization parameter lambda
%   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.
%

% Initialize Theta
initial_theta = zeros(size(X, 2), 1); 

% Create "short hand" for the cost function to be minimized
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Minimize using fmincg
theta = fmincg(costFunction, initial_theta, options);

end
%!test
%! X = [[1 1 1]' magic(3)];
%! y = [7 6 5]';
%! theta = [0.1 0.2 0.3 0.4]';
%! lambda = 0
%! [theta0] = trainLinearReg(X, y, lambda);
%! expected_theta0 = [0.078947; 0.478070; 0.228070; 0.478070;];
%! assert(theta0, expected_theta0, .0001);
