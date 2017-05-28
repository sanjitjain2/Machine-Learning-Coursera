function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);   % number of training examples
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);  %p is a (m x 1) column vector

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% X is (m x n) vector 
% all_theta is (num_labels x m) sized matrix i.e each row has theta for 
%                               respective classes
% z = (X * all_theta')
% h(theta) = sigmoid(z)               is a (m x num_labels) matrix means
%                                     each row has the values of h(theta)
%                                     for 10 labels out of which we have 
%                                     to find the maximum
% p = [m x (max of each row)]       here instead of storing the maximum of 
%                                   all h(theta) we need the index at which
%                                   maxima was found as that ud be the predicted 
%                                   digit 


z = (X * all_theta'); 
htheta = sigmoid(z);
[v p] = max(htheta,[],2);        %v has the values of max h(theta) and p has indexes



% =========================================================================


end
