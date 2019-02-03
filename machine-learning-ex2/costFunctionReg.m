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
t = theta;
t(1) = []; %Eliminating t0 for regularization. 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta); %Stores the hypothesis
temp = sum((-y.*log(h)) - ((1-y).*log(1-h)));
J1 = temp/m; %Cost of general term
J2 = (lambda * (t'*t))/(2*m); %Cost of regularized term
J = J1 + J2; %Total Cost

%Computing Gradient
grad1 = sum((h - y).*X)/m; %Gradient without Redularization
reg_term = (lambda/m) * theta; %Redularization Term Vector
grad = grad1' + reg_term; %Total Gradient
grad(1) -= reg_term(1); %Eliminating grad0 for regularization.
% =============================================================

end
