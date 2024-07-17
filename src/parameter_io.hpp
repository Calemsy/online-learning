#include <iostream>
#include <random>
#include <memory>
#include <map>
#include <numeric>
#include "tensor.hpp"


#define print_vector(param_name){											\
	size_t size = param_data[param_name].size();									\
	size_t cnt = 0;													\
	cout << param_name << ":[";											\
	for_each(param_data[param_name].begin(), param_data[param_name].end(), 						\
		[&cnt, size](const float& value) {cout << value << "],"[cnt++ != size - 1];}); 				\
	cout << endl;													\
}
#define LOGLEVEL 0
#define DLOG(args) \
if (LOGLEVEL == 1)cout << __FILE__ << ":" << __FUNCTION__ << ":" << __LINE__ << " [INFO] " << args << endl;

using namespace std;

class Parameter_IO {
public:
	Parameter_IO(const vector<string>&, const vector<vector<size_t>>&, const string&);
	void push(const std::vector<tensorflow::Tensor>&);
	void pull(std::vector<tensorflow::Tensor>&);
	void init(const string mode="uniform", const vector<float>&& para = {-1.0, 1.0});
	vector<float>& get_parameter_data(const string& parameter_name) {  // TOD
		return param_data[parameter_name];
	}
private:
	const vector<string> weight_bias_name_vec;
	map<string, vector<float> > param_data;  // vector<float>是右值
	const vector<vector<size_t>> weight_bias_shapes;
	const string save_weight_file;
};

Parameter_IO::Parameter_IO(const vector<string>& w_b_names, const vector<vector<size_t>>& wb_shapes, const string& save_path) 
	: weight_bias_name_vec(w_b_names),
	  weight_bias_shapes(wb_shapes), 
	  save_weight_file(save_path) {}

void Parameter_IO::init(const string mode, const vector<float>&& para) {
	random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(para[0], para[1]);
	std::vector<tensorflow::Tensor> init_weight_bias_tensor;
	for(size_t i = 0; i < weight_bias_shapes.size(); ++i) {
		auto total_size = accumulate(weight_bias_shapes[i].begin(), weight_bias_shapes[i].end(), 1, multiplies<size_t>());
		vector<float> random_init_weight;
		for(size_t i = 0; i < total_size; ++i) random_init_weight.push_back(dis(gen));
		init_weight_bias_tensor.push_back(make_tensor<float>(random_init_weight, weight_bias_shapes[i]));
	}
	push(init_weight_bias_tensor);
}

void Parameter_IO::push(const std::vector<tensorflow::Tensor>& param_tensor) {
	size_t cnt = 0;
	DLOG("push w_b_names:");
	for_each(weight_bias_name_vec.begin(), weight_bias_name_vec.end(), 
		[&param_tensor, &cnt](const string& para_name) { DLOG(para_name + ":" + std::to_string(param_tensor[cnt++].NumElements()) + ","); });
	cnt = 0;
	for_each(weight_bias_name_vec.begin(), weight_bias_name_vec.end(), 
			[this, &param_tensor, &cnt](const string& param_name) 
				{param_data.insert({param_name, vector<float>(param_tensor[cnt++].NumElements())});});
	DLOG(" total v: " + std::to_string(param_data.size()));
	for (size_t i = 0; i < weight_bias_name_vec.size(); ++i) {
		copy_n(param_tensor[i].flat<float>().data(), param_tensor[i].NumElements(), param_data[weight_bias_name_vec[i]].begin());
	}
	fstream fout(save_weight_file, fstream::out | fstream::binary);
	if(!fout) throw runtime_error("open save file error");
	for(auto iter = param_data.begin(); iter != param_data.end(); ++iter) {
		size_t param_name_size = iter->first.size(), param_size = iter->second.size();
		fout.write((char*)&param_name_size, sizeof(size_t));
		fout.write((char*)iter->first.c_str(), param_name_size);
		fout.write((char*)&param_size, sizeof(size_t));
		fout.write((char*)iter->second.data(), sizeof(float) * (iter->second.size()));
	}
	fout.close();
	// print_vector("b1");
}

void Parameter_IO::pull(std::vector<tensorflow::Tensor>& param_tensor) {
	param_data.clear();
	fstream fin(save_weight_file, fstream::in | fstream::binary);
	if(!fin) throw runtime_error("open load file error:" + save_weight_file);
	DLOG("load w_b_names:");
	while (!fin.eof()) {
		char ch = fin.peek();
		if (ch == -1) break;
		size_t param_name_size;
		fin.read((char*)&param_name_size, sizeof(size_t));
		char* param_name_buffer = new char[param_name_size + 1];
		fin.read((char*)param_name_buffer, param_name_size);
		param_name_buffer[param_name_size] = '\0';
		string param_name(param_name_buffer);
		delete []param_name_buffer;
		size_t param_size;
		fin.read((char*)&param_size, sizeof(size_t));
		DLOG(param_name + ":" + std::to_string(param_size));
		vector<float> params(param_size);
		fin.read((char*)params.data(), sizeof(float) * param_size);
		param_data.insert({param_name, params});	
	}
	fin.close();
	//print_vector("b1");

	for(size_t i = 0; i < weight_bias_shapes.size(); ++i) {
		param_tensor.push_back(make_tensor<float>(param_data[weight_bias_name_vec[i]], weight_bias_shapes[i]));
	}
}

