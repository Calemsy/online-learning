#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <map>
#include <unordered_map>
#include <numeric>
#include "parameter_pap.hpp"
#include "../utils/common.h"

using namespace std;

class ParameterBaseFile : public ParameterPP {
public:
	ParameterBaseFile(const string&, const vector<string>&, const vector<vector<size_t>>&, const string&);
	void push(const std::vector<tensorflow::Tensor>&) override;
	void pull(std::vector<tensorflow::Tensor>&) override;
	inline vector<float>& get_parameter_data(const string& parameter_name) { return param_data.at(parameter_name);}
private:
	unordered_map<string, vector<float>> param_data; // vector<float> is right-value
	const string save_weight_file;
};

ParameterBaseFile::ParameterBaseFile(const string& model_name, const vector<string>& w_b_names, const vector<vector<size_t>>& wb_shapes, const string& save_path) 
	: ParameterPP(model_name, w_b_names, wb_shapes), save_weight_file(save_path) {}

void ParameterBaseFile::push(const std::vector<tensorflow::Tensor>& param_tensor) {
	param_data.clear();
	size_t cnt = 0;
	// construct `param_tensor`
	OUTLOG("file_push: parameters name and flatten size");
	for_each(_parameter_name_vec.begin(), _parameter_name_vec.end(), 
		[this, &param_tensor, &cnt](const string& para_name) 
			{ 
				OUTLOG(para_name + ":" + TS(param_tensor[cnt].NumElements())); 
				param_data.insert({para_name, vector<float>(param_tensor[cnt].NumElements())});
				cnt++;
			});
	// copy from tf-tensor to c++-vector
	for (size_t i = 0; i < _parameter_name_vec.size(); ++i) {
		copy_n(param_tensor[i].flat<float>().data(), param_tensor[i].NumElements(), param_data.at(_parameter_name_vec[i]).begin());
	}
	// write to file
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
}

void ParameterBaseFile::pull(std::vector<tensorflow::Tensor>& param_tensor) {
	param_data.clear();
	// read from file
	fstream fin(save_weight_file, fstream::in | fstream::binary);
	if(!fin) throw runtime_error("open load file error:" + save_weight_file);
	OUTLOG("file_load: parameters name:");
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
		OUTLOG(param_name + ":" + TS(param_size));
		vector<float> params(param_size);
		fin.read((char*)params.data(), sizeof(float) * param_size);
		param_data.insert({param_name, params});	
	}
	fin.close();
	// copy from c++-vector to tf-tensor
	for(size_t i = 0; i < _parameters_shape.size(); ++i) {
		param_tensor.push_back(make_tensor<float>(param_data.at(_parameter_name_vec[i]), _parameters_shape[i]));
	}
}

