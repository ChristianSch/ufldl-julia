using Gadfly
using Optim: optimize

# loading linear regression functions
using linear_regression_vec
using linear_regression

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

m = size(train_X, 2)
n = size(train_X, 1)

# Initialize the coefficient vector theta to random values.
theta = rand(n)

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
# Compare the running time for your linear_regression.jl and
# linear_regression_vec.jl implementations.
#
# Uncomment the lines below to run your vectorized code.
# Re-initialize parameters
# theta = rand(n)
# tic() # start timer (is read by toc())
# result = optimize(J, g!, theta, method=:gradient_descent, iterations=200)
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
    layer(x=1:length(test_y), y=sort(test_y'[:, 1]), Geom.point),
    layer(x=1:length(test_y), y=sort(predicted_prices'[:, 1]), Geom.point),
)

