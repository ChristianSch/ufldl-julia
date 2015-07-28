using ex1_load_fns

binary_digits = true
data = ex1_load_mnist(binary_digits)

train_X = data[1][1]
train_y = data[1][2]

test_X = data[2][1]
test_y = data[2][2]
