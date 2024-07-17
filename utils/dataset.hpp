#pragma once
#include <arpa/inet.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>

using namespace std;

class dataset {
public:
	virtual void next_batch(vector<float>& x, vector<float>& y, const size_t batch_size = 1024) const = 0;
	inline const vector<float>& test_x() const {return test_x_vec;}
	inline const vector<float>& test_y() const {return test_x_vec;}

	const vector<float>& train_x() const = delete;
	const vector<float>& train_y() const = delete;
	virtual ~dataset() = default;
protected:
	vector<float> train_x_vec, test_x_vec, train_y_vec, test_y_vec;
};


class read_single_img {
public:
    	read_single_img(const string &path) : path(path) {}
    	size_t size() const {return data.size();}
    	vector<float> operator()() {
		ifstream fin(this->path);
		if (!fin)  throw runtime_error("open error" + this->path);
		string line, token;
		while (getline(fin, line)) {
			istringstream is(line);
			while (is >> token) {
				if (token == "[[" || token == "]]" || token == "[" || token == "]")
					continue;
				int pos;
				if((pos = token.find(',')) > 0) token.replace(pos, 1, "");
				if((pos = token.find("]]")) > 0) token.replace(pos, 2, "");
				data.push_back(atof(token.c_str()));
			}
		}
		for_each(data.begin(), data.end(), [](float &d) {d/=255.0;});
		fin.close();
		return data;
	}
private:
    	const string path;
    	vector<float> data;
};

struct image_norm {
public:
        image_norm(float fact) : fact(fact) {}
        void operator()(float &value) {
                value /= fact;
        }
private:
        float fact;
};
