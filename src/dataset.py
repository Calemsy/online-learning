import numpy as np
import gzip
import idx2numpy
from argparse import Namespace

print("load data start", end=",", flush=True)
mnist = Namespace()
with gzip.open('../data/mnist/t10k-images-idx3-ubyte.gz', 'rb') as f:
    mnist.test_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28) / 255.
with gzip.open('../data/mnist/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    mnist.test_y = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open('../data/mnist/train-images-idx3-ubyte.gz', 'rb') as f:
    mnist.train_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28) / 255.
with gzip.open('../data/mnist/train-labels-idx1-ubyte.gz', 'rb') as f:
    mnist.train_y = np.frombuffer(f.read(), np.uint8, offset=8)
print("mnist done", end=",", flush=True)

letters = Namespace()
letters.train_x = idx2numpy.convert_from_file('../data/letters/emnist-letters-train-images-idx3-ubyte')
letters.train_x = letters.train_x / 255.
letters.train_y = idx2numpy.convert_from_file('../data/letters/emnist-letters-train-labels-idx1-ubyte')
letters.test_x  = idx2numpy.convert_from_file('../data/letters/emnist-letters-test-images-idx3-ubyte')
letters.test_x  = letters.test_x / 255.
letters.test_y  = idx2numpy.convert_from_file('../data/letters/emnist-letters-test-labels-idx1-ubyte')
print("letters done", end=",", flush=True)

criteo = Namespace()
criteo.train_x = np.loadtxt("../data/criteo/train_x", dtype=int, delimiter=',')
criteo.train_y = np.loadtxt("../data/criteo/train_y", dtype=int, delimiter=',')
criteo.test_x  = np.loadtxt("../data/criteo/test_x", dtype=int, delimiter=',')
criteo.test_y  = np.loadtxt("../data/criteo/test_y", dtype=int, delimiter=',')
print("criteo done", end=",", flush=True)
print("load data end")
