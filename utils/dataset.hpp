#ifndef OL_DATASET
#define OL_DATASET

#include <arpa/inet.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>

using namespace std;
/*
 * for me, it's hard to support `dataset` as template class
template <typename T1, typename T2>
class dataset {
public:
        virtual void next_batch(vector<T1>& x, vector<T2>& y, const size_t batch_size = 1024) const = 0;
        virtual pair<vector<T1>, vector<T2>> next_batch(const size_t batch_size = 1024) const = 0;
        inline const vector<T1>& test_x() const {return test_x_vec;}
        inline const vector<T2>& test_y() const {return test_x_vec;}
        virtual ~dataset() = default;
protected:
        vector<T1> train_x_vec, test_x_vec;
        vector<T2> train_y_vec, test_y_vec;
};
*/
class dataset {
public:
	virtual pair<vector<float>, vector<float>> next_batch(const size_t batch_size = 1024) const = 0;
	inline const vector<float>& test_x() const {return test_x_vec;}
	inline const vector<float>& test_y() const {return test_x_vec;}
	virtual ~dataset() = default;
protected:
	vector<float> train_x_vec, test_x_vec;
	vector<float> train_y_vec, test_y_vec;
};
#endif
