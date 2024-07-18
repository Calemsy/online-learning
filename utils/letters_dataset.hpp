#include "xnist_dataset.hpp"

class letters_dataset : public xnist_dataset {
public:
    	letters_dataset() : xnist_dataset(124800) {
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
    	const string path = "../data/letters/";
    	const string train_x_path = path + "emnist-letters-train-images-idx3-ubyte"; 
    	const string train_y_path = path + "emnist-letters-train-labels-idx1-ubyte";
    	const string test_x_path  = path + "emnist-letters-test-images-idx3-ubyte";
    	const string test_y_path  = path + "emnist-letters-test-labels-idx1-ubyte";
};

