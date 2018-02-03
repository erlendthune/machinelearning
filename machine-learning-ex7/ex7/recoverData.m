function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               

X_rec = Z*U(:,1:K)';



% =============================================================

end
%!test
%! Z=[1.481274];
%! U = [
%!  -0.70711  -0.70711;
%!  -0.70711   0.70711;
%! ];
%! K=1;
%! X_rec  = recoverData(Z, U, K)
%! X_exp = [-1.047419 -1.047419];
%! assert(X_rec, X_exp, .0001);

%!test
%! Z = [
%!  -5.3519e+00   1.1650e+00   3.6637e-15;
%!  -1.3182e+01  -4.7297e-01  -1.9984e-15
%! ];
%! U = [
%!  -0.35206  -0.75898  -0.31752  -0.44630;
%!  -0.44363  -0.32124   0.75601   0.35840;
%!  -0.53519   0.11650  -0.55947   0.62209;
%!  -0.62675   0.55424   0.12097  -0.53420;
%! ];
%! K=3;
%! X_rec  = recoverData(Z, U, K);
%! X_exp = [1 2 3 4;5 6 7 8];
%! assert(X_rec, X_exp, .01);
