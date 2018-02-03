function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%
%For both C and sigma, we
%suggest trying values in multiplicative steps (e.g., 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30).
%Note that you should try all possible pairs of values for C and  (e.g., C = 0.3
%and sigma = 0:1).

min_error = 1;
theValues=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
for i = 1:rows(theValues)
    tempC = theValues(i)
    for j = 1:rows(theValues)
        tempSigma = theValues(j)
        model= svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma));
        predictions = svmPredict(model, Xval);
        new_error= mean(double(predictions ~= yval))
        if(new_error < min_error)
            min_error = new_error
            bestC = tempC;
            bestSigma = tempSigma;
        end
    endfor
endfor

C = bestC;
sigma = bestSigma;


% =========================================================================

end
%!test
%! X = [1 2; 3 4];
%! y = [0; 1];
%! Xval = [1 2; 3 4];
%! yval = [0; 1];
%! [C, sigma] = dataset3Params(X, y, Xval, yval)
%! expected_C = 0.01;
%! assert(C, expected_C, .0001);
%! expected_sigma = 0.01;
%! assert(sigma, expected_sigma, .0001);
