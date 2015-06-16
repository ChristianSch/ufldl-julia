using Gadfly
using Optim: optimize

# load the data from file
data = readdlm("housing.data")

# put examples in columns
data = data'

# Include a row of 1s as an additional intercept feature.
data = [ ones(1, size(data, 2)); data ]

# Shuffle examples.
data = data[:, randperm(size(data, 2)) ]

# training data
# X - The examples stored in a matrix.
#     X(i,j) is the i'th coordinate of the j'th example.
train_X = data[1:end-1, 1:400]

# y - The target value for each example.
# where y(j) is the target for example j.
train_y = data[end, 1:400]

# testing data
test_X = data[1:end-1,401:end]
test_y = data[end,401:end]

# number of training data sets
m = size(train_X, 2)

# number of features
n = size(train_X, 1)

# Initialize the coefficient vector theta to random values.
theta = rand(n)

### These functions were originally defined in 'linear_regression.jl'
# TODO:  Implement the linear regression objective and gradient computations
#        The linear regression objective is J and the gradient is g!
#
# Note: originally named `linear_regression` I find that J for the error
# function (or objective function) is much clearer.
function J(_theta::Vector)
    obj = 0

    # TODO:  Compute the linear regression objective by looping over the examples in X.
    #        Store the objective function value in 'obj' (originaly 'f').

    ### YOUR CODE HERE ###
end

# other than the matlab version of this excercise,
# we calculate the gradient in another function due to
# the optimize function of the Optim package.
function g!(theta::Vector, storage::Vector)
    for k = 1:length(storage)
        storage[k] = 0
    end

    # TODO:  Compute the gradient of the objective with respect to theta by looping over
    #        the examples in X and adding up the gradient for each example.  Store the
    #        computed gradient in 'storage' ('g' in the original assignment).

    ### YOUR CODE HERE ###
end

### These methods were originally defined in linear_regression_vec.jl
# TODO:  Implement the linear regression objective and gradient computations
#        The linear regression objective is J and the gradient is g!
#
# Note: originally named `linear_regression` I find that J for the error
# function (or objective function) is much clearer.
function J_vec(theta::Vector)
    obj = 0

    # TODO:  Compute the linear regression objective by looping over the examples in X.
    #        Store the objective function value in 'obj' (originaly 'f').

    ### YOUR CODE HERE ###
    for i = 1:m
        # this results in an array with one element, thus we need
        # to take the first (and only) element
        obj += ((theta' * train_X[:, i]) - train_y[i])[1]^2
    end

    println("J: $(obj/2)")

    obj / 2
end

# Run the optimize function with J (originally 'linear_regression') as the objective.
#
# TODO:  Implement linear regression in linear_regression.jl

tic() # start timer (is read by toc())
result = optimize(J, g!, theta, method=:gradient_descent, iterations=200)

elapsed = toc()
println("Optimization took $elapsed seconds.")

# Run the optimize function with J (originally 'linear_regression') as the objective.
#
# TODO:  Implement linear regression in linear_regression_vec.jl
# using vectorization features to speed up your code (certainly not true
# for julia).
# Compare the running time for your linear_regression and
# linear_regression_vec implementations.
#
# Uncomment the lines below to run your vectorized code.
# Re-initialize parameters
# theta = rand(n)
# tic() # start timer (is read by toc())
# result = optimize(J_vec, g!, theta, method=:gradient_descent, iterations=200)
#
# elapsed = toc()
# println("Optimization took $elapsed seconds.")


# Root-Mean-Squared training error
train_rms = sqrt(mean(((result.minimum' * train_X) - train_y).^2))
println("Train RMS: $train_rms")

# Root-Mean_Squared test error
test_rms = sqrt(mean(((result.minimum' * test_X) - test_y).^2))
println("Test RMS: $test_rms")

Gadfly.plot(
    Guide.xlabel("House #"), Guide.ylabel("House price (\$1000s)"),
    layer(x=1:length(test_y), y=sort(test_y'[:, 1]), Geom.point, Theme(default_color=color("red"))),
    layer(x=1:length(test_y), y=sort((result.minimum' * test_X)'[:, 1]), Geom.point),
)

