function [theta, cost] = advancedoptimization(X, y, initial_theta)
options = optimset('GradObj', 'on', 'MaxIter', 400);
   [theta, cost] = fminunc(@(t) (costFunction(t, X, y)), initial_theta, options);
end