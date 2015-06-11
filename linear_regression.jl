module linear_regression
export J
export g!

# TODO:  Implement the linear regression objective and gradient computations
#        The linear regression objective is J and the gradient is g!
#
# Note: originally named `linear_regression` I find that J for the error
# function (or objective function) is much clearer
function J(_theta::Vector)
    obj = 0

    # TODO:  Compute the linear regression objective by looping over the examples in X.
    #        Store the objective function value in 'obj' (originally 'f').

    ### YOUR CODE HERE ###
end

# other than the matlab version of this excercise,
# we calculate the gradient in another function due to
# the optimize function of the Optim package.
function g!(_theta::Vector, storage::Vector)
    storage = zeros(length(storage))
    # TODO:  Compute the gradient of the objective with respect to theta by looping over
    #        the examples in X and adding up the gradient for each example.  Store the
    #        computed gradient in 'storage' ('g' in the original assignment).

    ### YOUR CODE HERE ###
end

end
