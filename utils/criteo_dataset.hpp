#include<iostream>
#include<fstream>
#include"dataset.hpp"
#include"common.h"

class criteo_dataset : public dataset {
public:
	criteo_dataset() {
		std::ifstream fin(file_name);
		if (!fin) {
			throw std::runtime_error(file_name + std::string(" not exists."));
		}
		std::vector<float> hid;
		std::vector<float> labels;
		std::string line;
		while (getline(fin, line)) {
			auto sample = stringSplit(line, '\t');
			for(size_t i = 0; i < sample.size(); ++i) {
				if (i == 0) {
					labels.push_back(std::atof(sample.at(0).c_str()));
				} else {
					hid.push_back(get_fid(i, sample.at(i)));
				}
			}
			lineno++;
		}
		size_t split_y = int(lineno * 0.90);
		size_t split_x = feature_size * split_y;
		OUTLOG(TS2(split_y) + ", " + TS2(split_x) + ", " + TS2(hid.size()) + ", " + TS2(labels.size()))
		train_x_vec.assign(hid.begin(), hid.begin() + split_x);
		train_y_vec.assign(labels.begin(), labels.begin() + split_y);
		test_x_vec.assign(hid.begin() + split_x, hid.end());
		test_y_vec.assign(labels.begin() + split_y, labels.end());
		OUTLOG(TS2(train_x_vec.size()) + ", " + TS2(train_y_vec.size()) + ", " + TS2(test_x_vec.size()) + ", " + TS2(test_y_vec.size()))
		fin.close();
	}
	pair<vector<float>, vector<float>> next_batch(const size_t batch_size = 1024) const override;
	void write2file() const;
private:
	mutable int count = 0;
	size_t lineno = 0;
	const char* file_name = "../data/criteo/criteo_demo.txt";
	static constexpr int feature_size = 39;
};

pair<vector<float>, vector<float>> criteo_dataset::next_batch(const size_t batch_size) const {
	if (++count > int(lineno / batch_size)) {
		OUTLOG("all criteo data has been trained.");
		count = 0;
	}
	vector<float> x, y;
	x.insert(x.end(), train_x_vec.begin() + count * feature_size * batch_size, train_x_vec.begin() + (count + 1) * feature_size * batch_size);
	y.insert(y.end(), train_y_vec.begin() + count * batch_size, train_y_vec.begin() + (count + 1) * batch_size);
	return make_pair(move(x), move(y));
}

void criteo_dataset::write2file() const {
	auto write_x = [](const vector<float>& vx, const string& file_name) {
		std::ofstream fout(file_name);
		if (!fout) {
			throw std::runtime_error(file_name + std::string("create error."));
		}
		for(size_t i = 0; i < vx.size(); ++i) {
			fout << vx.at(i);
			if ((1 + i) % 39 == 0 && i != 0) {
				fout << "\n";
			} else {
				fout << ",";
			}
		}
		fout.close();
	};
	auto write_y = [](const vector<float>& vy, const char* file_name) {
		std::ofstream fout(file_name);
		if (!fout) {
			throw std::runtime_error(file_name + std::string("create error."));
		}
		for(size_t i = 0; i < vy.size() - 1; ++i) {
			fout << vy.at(i) << ",";
		}
		fout << vy.at(vy.size() - 1);
		fout.close();
	};
	write_x(train_x_vec, "../data/criteo/train_x");
	write_y(train_y_vec, "../data/criteo/train_y");
	write_x(test_x_vec,  "../data/criteo/test_x");
	write_y(test_y_vec,  "../data/criteo/test_y");
}

