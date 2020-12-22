# Getting MNIST data sets

```
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

# Decompress

```
gzip -d *.gz
```

# Place

Place the files in the path you choose.

The software will accept command line arguments for the files to
read for training.


