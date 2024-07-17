import numpy as np
import gzip
import idx2numpy
from argparse import Namespace

mnist = Namespace()
with gzip.open('../data/mnist/t10k-images-idx3-ubyte.gz', 'rb') as f:
    mnist.test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
with gzip.open('../data/mnist/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    mnist.test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open('../data/mnist/train-images-idx3-ubyte.gz', 'rb') as f:
    mnist.train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
with gzip.open('../data/mnist/train-labels-idx1-ubyte.gz', 'rb') as f:
    mnist.train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

letters = Namespace()
letters.train_images = idx2numpy.convert_from_file('../data/letters/emnist-letters-train-images-idx3-ubyte')
letters.train_labels = idx2numpy.convert_from_file('../data/letters/emnist-letters-train-labels-idx1-ubyte')
letters.test_images = idx2numpy.convert_from_file('../data/letters/emnist-letters-test-images-idx3-ubyte')
letters.test_labels = idx2numpy.convert_from_file('../data/letters/emnist-letters-test-labels-idx1-ubyte')
