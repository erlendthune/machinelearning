function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X*theta);
thetareg = theta(2:size(theta))
Xreg = X(1:size(X),2:3)

r = lambda / (2 * m) * sum(thetareg.^2)

%Another solution is to calculate r (p below) like this:
%theta1 = [0 ; theta(2:size(theta), :)];
%p = lambda*(theta1'*theta1)/(2*m);

J = 1 / m * (-y' * log(h) - (1-y)'*log(1-h)) + r

%Another solution is to calculate J like this. Q
%J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + p;

%grad0 is correct
grad0 = 1 / m * ((h-y)' * X(:,1))

%Remeber that the sum sign is automatically done when multiplying matrices.
%Also remember that you might have to transpose a matrix before adding it to another.
grad1andon = 1 / m * ((h-y)' * Xreg)' + lambda/m*thetareg

grad = [grad0; grad1andon]

%A much more elegant solution
%grad = (X'*(h - y)+lambda*theta1)/m;


% =============================================================

end

