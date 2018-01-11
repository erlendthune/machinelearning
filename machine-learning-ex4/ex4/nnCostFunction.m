function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   
disp("IN Input layer size:"), disp(input_layer_size)
disp("IN Hidden layer size:"), disp(hidden_layer_size)
disp("IN Number of labels:"), disp(num_labels)
disp("IN X size:"), disp(size(X))
                                   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Include the bias
s1 = input_layer_size;
s2 = hidden_layer_size;
s3 = num_labels;
tVec=nn_params;
         
t1Elems=(s1+1)*s2; %+1 is for the bias
t2Elems=(s2+1)*s3;

t2StartIndex=t1Elems+1;
t2EndIndex=t1Elems+t2Elems;

Theta1=reshape(tVec(1:t1Elems),s2,s1+1);
Theta2=reshape(tVec(t2StartIndex:t2EndIndex),s3,s2+1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Y=eye(num_labels);
X = [ones(m, 1) X];
disp("X size:"), disp(size(X))
for i = 1:m
	a1 = X(i,:);
	%Compute h
	z2=Theta1*a1';
	a2=sigmoid(z2);
	a2 = [1 a2'];
	z3=Theta2*a2';
	a3=sigmoid(z3);
	h=a3';
	
	yi=Y(y(i),:);
    J = J - yi * log(h)' - (1-yi)*log(1-h)';
    
    delta3=a3-yi';
    
    %Need to add bias node for activation layer
    z2 = [1 ; z2];
    delta2=(Theta2' * delta3) .* sigmoidGradient(z2);
    delta2=delta2(2:end);
    
    Theta2_grad = Theta2_grad + delta3*a2;
%    disp("Size a1"),disp(size(a1))
%    disp("Size delta2"), disp(size(delta2))
%    disp("Size Theta1_grad"), disp(size(Theta1_grad))

    Theta1_grad = Theta1_grad + delta2*a1;
end

J = J / m; 

% add regularization to cost function
Theta1_reg = Theta1(:,2:size(Theta1)(2));
r1_a = Theta1_reg.^2;
r1_b = sum(r1_a);
r1 = sum(r1_b);

disp("r1:"), disp(r1)

Theta2_reg = Theta2(:,2:size(Theta2)(2));
r2_a = Theta2_reg.^2;
r2_b = sum(r2_a);
r2 = sum(r2_b);

disp("r2:"), disp(r2)

r = lambda/(2*m)*(r1 + r2);

J = J + r

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
