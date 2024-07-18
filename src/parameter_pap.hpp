#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "tensor.hpp"
#include "../utils/common.h"

class ParameterPP {
// `PP` is mean Push and Pull
public:
	ParameterPP(const std::string& model_name, const std::vector<std::string>& parameter_name_vec, const std::vector<vector<size_t>>& parameters_shape) 
		: _model_name(model_name)
		, _parameter_name_vec(parameter_name_vec)
		, _parameters_shape(parameters_shape) {}
	virtual void push(const std::vector<tensorflow::Tensor>&) = 0;
	virtual void pull(std::vector<tensorflow::Tensor>&) = 0;
	virtual void init_parameters(const string&, const vector<float>&&);
	virtual ~ParameterPP() = default;
protected:
	const std::string _model_name;
	const std::vector<std::string> _parameter_name_vec;
	const std::vector<vector<size_t>> _parameters_shape;
};

void ParameterPP::init_parameters(const string& mode="uniform", const vector<float>&& para={-0.1, 0.1}) {
        random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(para[0], para[1]);
        std::vector<tensorflow::Tensor> init_weight_bias_tensor;
        for(size_t i = 0; i < _parameters_shape.size(); ++i) {
                auto total_size = accumulate(_parameters_shape[i].begin(), _parameters_shape[i].end(), 1, multiplies<size_t>());
                vector<float> random_init_weight(total_size);
		std::generate(random_init_weight.begin(), random_init_weight.end(), [&dis, &gen](){return dis(gen);});
		OUTLOG("init dense name:" + _parameter_name_vec.at(i) + ", size:" + TS(total_size))
		CHECK_EQ(total_size, random_init_weight.size());
                init_weight_bias_tensor.push_back(make_tensor<float>(random_init_weight, _parameters_shape[i]));
        }
        push(init_weight_bias_tensor);
}
