module ex1_load_fns
export loadMNISTImages, loadMNISTLabelsl, ex1_load_mnist

function loadMNISTImages(filename::String)
    fp = open(filename, "r");

    magic = hton(read(fp, Int32))
    @assert(magic == 2051, "Bad magic number in $filename")

    numImages = hton(read(fp, Int32))
    println("numImages: $numImages")
    numRows = hton(read(fp, Int32))
    println("numRows: $numRows")
    numCols = hton(read(fp, Int32))
    println("numCols: $numCols")

    images = read(fp, Int8, numRows * numCols * numImages)
    # convert to big endian
    images = map(hton, images)

    images = reshape(images,
        convert(Int64, numCols),
        convert(Int64, numRows),
        convert(Int64, numImages))

    images = permutedims(images, [2 1 3])
    close(fp)

    # reshape to #pixel x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3))

    # convert to double and rescale to [0,1]
    images = images ./ 255.0

    return images
end

function loadMNISTLabels(filename::String)
    fp = open(filename, "r");

    magic = hton(read(fp, Int32))
    @assert(magic == 2049, "Bad magic number in $(filename)")

    numLabels = hton(read(fp, Int32))
    labels = read(fp, Int8, numLabels)
    # convert to big endian
    labels = map(hton, labels)

    # this shouldn't be possible anyway ...
    @assert(size(labels, 1) == numLabels, "Mismatch in label count")

    close(fp)

    return labels
end

function ex1_load_mnist(binary_digits)
    # images
    X = loadMNISTImages("common/train-images-idx3-ubyte")
    # labels
    y = loadMNISTLabels("common/train-labels-idx1-ubyte")

    if (binary_digits)
        # take only the 1 and 0 digits
        # X = [ X[ :, y == 0 ], X[ :, y == 1 ] ] # <- original solution, does not work
        X = filter(x -> x == 0 || x == 1, X) # not working
        # y = [ y[ y == 0 ], y[ y == 1 ] ] # <- original solution, does not work
        y = filter(y_ -> y_ == 0 || y == 1, y) # not working
        # note: not working either:
        # [X[:, y.==1], X[:, y.==0]]
    end

    # Randomly shuffle the data
    # genrate random numbers from 1 to length of the vector
    # to have random indices
    I = randperm(length(y))

    y = y[I]
    X = X[:, I]

    #  We standardize the data so that each pixel will
    # have roughly zero mean and unit variance.
    s = std(X, 2)
    m = mean(X, 2)
    X = broadcast(-, X, m)
    X = broadcast(./, X, s .+ 1)

    # Place these in the training set
    train_X = X
    train_y = y

    # Load the testing Data
    # images
    X = loadMNISTImages("common/train-images-idx3-ubyte")
    # labels
    y = loadMNISTLabels("common/t10k-labels-idx1-ubyte")

    if (binary_digits)
        # take only the 1 and 0 digits
        # X = [ X[ :, y == 0 ], X[ :, y == 1 ] ] # <- original solution, does not work
        X = filter(x -> x == 0 || x == 1, X)
        # y = [ y[ y == 0 ], y[ y == 1 ] ] # <- original solution, does not work
        y = filter(y_ -> y_ == 0 || y == 1, y)
    end

    # Randomly shuffle the data
    I = randperm(length(y))
    y = y[ I ]
    X = X[ :, I ]

    # Standardize using the same mean and scale as the training data.
    X = broadcast(-, X, m)
    X = broadcast(./, X, s .+ 1)

    # Place these in the testing set
    test_X = X
    test_y = y

    return [[train_X, train_y], [test_X, test_y]]
end

end
