module ex1_load_mnist
export loadMNISTImages, loadMNISTLabels

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

    images = reshape(images, 28, 28, 10000) # numCols, numRows, numImages
    images = permutedims(images,[2 1 3])
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

end
