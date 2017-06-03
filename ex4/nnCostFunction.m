function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
%----------------------------Part 1 Implementation---------------------------

y_matrix= zeros(m,num_labels); %Expand the 'y' output values into a matrix of single values

for i = 1:m
    y_matrix(i,y(i)) = 1;
end 

X = [ones(m,1) X];      %add a column of ones i.e. BIAS UNIT to X

z2 = X*Theta1';         %z2 is (5000 x 25) matrix
a2 = sigmoid(z2);       %compute sigmoid of z2

a2 = [ones(m,1) a2];    %add a column of ones i.e BASI UNIT to a2

z3 = a2*Theta2';        %z3 is (5000 x 10) matrix
a3 = sigmoid(z3);       %compute sigmoid of z3

htheta = a3;            %hypothesis or hthetaX obtained from forward propag.

J = (-1/m) * (sum(sum(y_matrix .* log(htheta) + (1 - y_matrix) .* log(1 - htheta))));       % costfuntionJ without regularization using two summations

regularized_term = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));      %regularized term
  
J = J + regularized_term;       %adding it to the costfunctionJ

%----------------------------Part 2 Implementation---------------------------

for t = 1:m
    a1 = [X(t,:)'];    %for input layer where l=1[1;a] adds 1 to a for bias units
    
    z2 = Theta1 * a1;    %for hidden layers where l=2
    a2 = [1;sigmoid(z2)];% add bias units   
    
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    y_k = ([1:num_labels]==y(t))';  %create column vector from 1:num_labels
                                    %with simaltaneous checking with Y.

    d3 = a3 - y_k;
    
    d2 = (Theta2' * d3) .* [1;sigmoidGradient(z2)];     %computing d2 and also adding a bias 
                                                                %unit bcos Theta2 and others have bias unit so needed for multiplication
        
    d2 = d2(2:end);    %removing the bias unit as we dont calc errors in bias unit
    
    Theta1_grad = Theta1_grad + d2 * a1';
    Theta2_grad = Theta2_grad + d3 * a2'; 
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
