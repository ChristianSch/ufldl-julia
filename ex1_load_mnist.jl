module ex1_load_mnist
export loadMNISTImages

function loadMNISTImages(filename::String)
    fp = open(filename, "r");

    magic = hton(read(fp, Uint32))
    @assert(magic == 2051, "Bad magic number in $filename")

    numImages = hton(read(fp, Uint32))
    println("numImages: $numImages")
    numRows = hton(read(fp, Uint32))
    println("numRows: $numRows")
    numCols = hton(read(fp, Uint32))
    println("numCols: $numCols")

    images = read(fp, Uint8, numRows * numCols * numImages)

    images = reshape(images, 28, 28, 10000) # numCols, numRows, numImages
    images = permutedims(images,[2 1 3])
    close(fp)

    # reshape to #pixel x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3))

    # convert to double and rescale to [0,1]
    images = images ./ 255.0

    return images
end

end
