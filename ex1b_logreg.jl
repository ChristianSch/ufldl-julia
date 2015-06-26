using ex1_load_mnist

images = loadMNISTImages("common/t10k-images-idx3-ubyte")
println("sizeof images: $(size(images, 1)) x $(size(images, 2))")
