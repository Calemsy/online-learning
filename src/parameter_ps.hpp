#include <iostream>
#include <cstring>
#include <numeric>
#include "parameter_pap.hpp"
#include "ps/ps.h"
#include "../utils/common.h"

using namespace ps;

class ParameterBasePS : public ParameterPP {
public:
	ParameterBasePS(const std::string& model_name, const std::vector<std::string>& parameter_name_vec, const std::vector<vector<size_t>>& parameters_shape, size_t worker_id, size_t model_ps_key_offset, int total_worker_num) 
		: ParameterPP(model_name, parameter_name_vec, parameters_shape)
		, _worker_id(worker_id)
		, _model_ps_key_offset(model_ps_key_offset)
		, _total_worker_num(total_worker_num)
	{
		init();
		for(const auto& shape : _parameters_shape) {
			_parameters_lens.push_back(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
		}
		_ps_keys.resize(_parameters_lens.size(), 0);
		std::iota(_ps_keys.begin(), _ps_keys.end(), _model_ps_key_offset);
		_split_pull_result.resize(_parameters_lens.size() + 1, 0);
		std::partial_sum(_parameters_lens.begin(), _parameters_lens.end(), _split_pull_result.begin() + 1);
		OUTLOG("Start ps-client:" + std::to_string(_worker_id))
		Start(_worker_id);
		kv_worker = new KVWorker<float>(0, _worker_id);
	}
	
	void push(const std::vector<tensorflow::Tensor>&) override;
	void pull(std::vector<tensorflow::Tensor>&) override;
	void init() const;

	~ParameterBasePS() {
		delete kv_worker;
		Finalize(_worker_id, true);
	}

private:
	std::vector<size_t> _parameters_lens;
	std::vector<size_t> _split_pull_result;
	map<string, vector<float> > param_data; 
	uint32_t _model_ps_key_offset;
	std::vector<Key> _ps_keys;
	const int _total_worker_num;
	KVWorker<float>* kv_worker;
	const int _worker_id;
};

void ParameterBasePS::push(const std::vector<tensorflow::Tensor>& grad_tensor) {
	param_data.clear();
        size_t cnt = 0;
        for_each(_parameter_name_vec.begin(), _parameter_name_vec.end(),
                [this, &grad_tensor, &cnt](const string& para_name)
                        {
                                param_data.insert({para_name, vector<float>(grad_tensor[cnt].NumElements())});
				cnt++;
                        });
        // copy from tf-tensor to c++-vector for push
	std::vector<float> vals;
	std::vector<int> lens;
        for (size_t i = 0; i < _parameter_name_vec.size(); ++i) {
		auto num_elements = grad_tensor[i].NumElements();
		auto& val = param_data.at(_parameter_name_vec.at(i));
                copy_n(grad_tensor[i].flat<float>().data(), num_elements, val.begin());
		vals.insert(vals.end(), val.begin(), val.end());
		lens.push_back(num_elements);
        }
	// ps-lite push
        int repeat = 1;
        std::vector<int> ts;
        for (int i = 0; i < repeat; ++i) {
                ts.push_back(kv_worker->Push(_ps_keys, vals, lens));
        }
        for (int t : ts) kv_worker->Wait(t);
}

void ParameterBasePS::pull(std::vector<tensorflow::Tensor>& param_tensor) {
	param_data.clear();
	// pull from ps-lite
        std::vector<float> rets;
        kv_worker->Wait(kv_worker->Pull(_ps_keys, &rets));
	if (rets.size() == 0) {
		OUTLOG("New model need to be initializer")
		init_parameters("uniform", {-0.1, 0.1});
		kv_worker->Wait(kv_worker->Pull(_ps_keys, &rets));
	}
	// fill into `param_data` according to `_split_pull_result`
	for(size_t i = 0; i < _parameter_name_vec.size(); ++i) {
		size_t start = _split_pull_result.at(i), stop = _split_pull_result.at(i + 1);
		const vector<float> parameter(rets.begin() + start, rets.begin() + stop);
		param_data.insert({_parameter_name_vec.at(i), parameter});
	}
	// from c++-vector to tf-tensor
	for(size_t i = 0; i < _parameter_name_vec.size(); ++i) {
		param_tensor.push_back(make_tensor<float>(param_data[_parameter_name_vec.at(i)], _parameters_shape.at(i)));
	}
}

void ParameterBasePS::init() const {
	const std::unordered_map<std::string, std::string> PS_ENV = {
                {"DMLC_NUM_SERVER", 	"1"}, // TODO
                {"DMLC_NUM_WORKER", 	std::to_string(_total_worker_num).c_str()},
                {"DMLC_PS_ROOT_URI", 	"127.0.0.1"},
                {"DMLC_PS_ROOT_PORT", 	"8111"},
		{"DMLC_ROLE", 		"worker"}
        };
	for(const auto& config : PS_ENV) {
                setenv(config.first.c_str(), config.second.c_str(), 1);
        }
	char heapprofile[6] = "./W";
	strcat(heapprofile, std::to_string(_total_worker_num - 1).c_str());
	setenv("HEAPPROFILE", heapprofile, 1);
	OUTLOG(string("start work ") + heapprofile)
}
