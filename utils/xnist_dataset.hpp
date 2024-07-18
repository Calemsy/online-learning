#pragma once
#include "dataset.hpp"
#include "common.h"
#include <algorithm>

class xnist_dataset : public dataset {
public:
    	xnist_dataset(const int total_train_x): total(total_train_x) {}

        virtual pair<vector<float>, vector<float>> next_batch(const size_t batch_size = 1024) const override {
                random_device rd;  // Will be used to obtain a seed for the random number engine
        	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        	std::uniform_real_distribution<> dis(0, total - 1);
                vector<unsigned> ret(batch_size);
		std::generate(ret.begin(), ret.end(), [&dis, &gen](){return dis(gen);});
		vector<float> x, y;
                for (const auto r : ret) {
                        x.insert(x.end(), train_x_vec.begin() + r * rows * cols, train_x_vec.begin() + (r + 1) * rows * cols);
                        y.push_back(train_y_vec[r]);
                }
		return make_pair(move(x), move(y));
        }

	virtual ~xnist_dataset() = default;

protected:
    	vector<float> read_x(const string& path) {
        	ifstream fin(path, fstream::in);
        	if (!fin) throw runtime_error("can't read file path:" + path);
        	if (_read_uint32(fin) != 2051) throw runtime_error("magic parse error");
		uint32_t num = _read_uint32(fin);
		uint32_t rows = _read_uint32(fin);
		uint32_t cols = _read_uint32(fin);
		vector<float> image;
		for(uint32_t i = 0; i < num * rows * cols; ++i) {
            		image.push_back(_read_uint8(fin));
        	}
		OUTLOG("read: " + path + " done, size " + std::to_string(image.size()))
		this->num = num; 
		this->rows = rows;
		this->cols = cols;
		return std::move(image);
    	}
    	vector<float> read_y(const string& path) {
		ifstream fin(path, fstream::in);
		if (!fin) throw runtime_error("can't read file y");
		if (_read_uint32(fin) != 2049) throw runtime_error("magic parse error");
		uint32_t num = _read_uint32(fin);
		vector<float> label;
		for (uint32_t i = 0; i < num; ++i) {
		 	label.push_back(_read_uint8(fin));
		}
		OUTLOG("read: " + path + " done, size " + std::to_string(label.size()))
		return std::move(label);
    	}
    	uint32_t _read_uint32(ifstream& is) {
		uint32_t data = 0;
		if (is.read(reinterpret_cast<char*>(&data), 4)) {
		    return ntohl(data);
		}
		throw runtime_error("can't read 4 bytes");
    	}
    	uint8_t _read_uint8(ifstream& is) {
		uint8_t data = 0;
		if (is.read(reinterpret_cast<char*>(&data), 1)) {
		    return data;
		}
		throw runtime_error("can't read 1 byte");
    	}

    	uint32_t num, rows, cols;
	const int total;
};

