function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X = [ones(m,1) X];
h = X*theta; %Hypothesis obtained for each data points
temp = sum((h - y).^2); %Sum of Squared Differences
t = theta;
t(1) = [];
J = (1/(2*m)) * (temp + (lambda * sum(t.^2))); % Regularized Cost Function
% =========================================================================

%Computing Gradient
grad1 = sum((h - y).*X)/m; %Gradient without Redularization
grad1 = grad1(:);
reg_term = (lambda/m) * theta; %Redularization Term Vector
grad = grad1 + reg_term; %Total Gradient
grad(1) -= reg_term(1); %Eliminating grad0 for regularization.
grad = grad(:);

end
