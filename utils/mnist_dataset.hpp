#include "xnist_dataset.hpp"

class mnist_dataset : public xnist_dataset {
public:
    	mnist_dataset() : xnist_dataset(60000) {
        	train_x_vec = read_x(train_x_path);
        	train_y_vec = read_y(train_y_path);
		test_x_vec = read_x(test_x_path);
		test_y_vec = read_y(test_y_path);
		for_each(train_x_vec.begin(), train_x_vec.end(), [](float &x){x /= 255.0;});
		for_each(test_x_vec.begin(), test_x_vec.end(), image_norm(255.0));
		for_each(train_y_vec.begin(), train_y_vec.end(), image_norm(1.0));
		for_each(test_y_vec.begin(), test_y_vec.end(), image_norm(1.0));
	}

private:
    	const string path = "../data/mnist/";
    	string train_x_path = path + "train-images-idx3-ubyte";
    	string train_y_path = path + "train-labels-idx1-ubyte";
    	string test_x_path = path + "t10k-images-idx3-ubyte";
    	string test_y_path = path + "t10k-labels-idx1-ubyte";
};

