#include "ol_trainer.h"

namespace tol {

template<typename T1, typename T2> vector<float> Trainer<T1, T2>::train_online(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data) {
	if (ps_backend == PS) [[likely]] {
		return train_online_base_ps(batch_x_data, batch_y_data);
	} else if (ps_backend == IO) {
		return train_online_base_file(batch_x_data, batch_y_data);
	} else {
		OUTERR("ps_backend error")
		return {};
	}
}

template<typename T1, typename T2> vector<float> Trainer<T1, T2>::train_online_base_ps(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data) {
	struct timeval begin, end;
	gettimeofday(&begin, 0);
	tensor_vector inputs;
	/**
	* virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
        *            	     const std::vector<string>& output_tensor_names,
        *            	     const std::vector<string>& target_node_names,
        *                    std::vector<Tensor>* outputs) = 0;
	* Runs the graph with the provided `inputs` tensors and fills `outputs` for the endpoints specified in `output_tensor_names`. 
	* Runs to but does not return Tensors for the nodes in `target_node_names`. 
	* The order of tensors in `outputs` will match the order provided by `output_tensor_names`.
	* If `Run` returns `OK()`, then `outputs->size()` will be equal to `output_tensor_names.size()`. If `Run` does not return
	*  `OK()`, the state of `outputs` is undefined.
	*
	* REQUIRES: The name of each Tensor of the input or output must match a "Tensor endpoint" in the `GraphDef` passed to `Create()`.
	* REQUIRES: At least one of `output_tensor_names` and  `target_node_names` must be non-empty.
	* REQUIRES: outputs is not nullptr if `output_tensor_names` is non-empty.
 	*/
	TF_CHECK_OK(run_env->session->Run(inputs, {}, {"ws_init"}, nullptr));
	// fill input `x` and `y`
	MakeIOTensor<T1, T2> data_set2tensor(batch_x_data, mc->get_input_size(), batch_y_data, mc->get_output_size());
	// name `input_x` and `input_y` is identity in tensorflow graph
	inputs = {{"input_x", data_set2tensor.x}, {"input_y", data_set2tensor.y}};
	
	// pull parameters
	std::vector<tensorflow::Tensor> wb_tensor;
	param_handler->pull(wb_tensor);
	const std::vector<std::string>& dense_name_vec = mc->get_dense_name_vec();
	for(size_t i = 0; i < wb_tensor.size(); ++i) {
		// on the basis of the original training samples x(input_x) and y(input_y) in `inputs`,
		// add the weights(weight and bias) pulled from the parameter server.
		inputs.push_back({dense_name_vec.at(i), wb_tensor.at(i)});
	}

	// run gradient
	const vector<string>& grad_op_vec = mc->get_gradient_name_vec();
	std::vector<tensorflow::Tensor> gradient_outputs;
	TF_CHECK_OK(run_env->session->Run(inputs, grad_op_vec, {}, &gradient_outputs));

	// push gradient
	param_handler->push(gradient_outputs);

	// calculate metrics (loss, accuracy, etc.) using the updated parameters by `batch_x_data` and `batch_y_data`
	// 1、erase old `parameters` from `inputs`
	inputs.erase(inputs.end() - dense_name_vec.size(), inputs.end());
	// 2、pull new parameter
	wb_tensor.clear();
	param_handler->pull(wb_tensor);
	// 3、add new parameter into `inputs`
	for(size_t i = 0; i < wb_tensor.size(); ++i) {
		inputs.push_back({dense_name_vec.at(i), wb_tensor.at(i)});
	}
	
        // run observe metrics
        std::vector<tensorflow::Tensor> metrics;
        const vector<string>& log_vec = mc->get_observe_name_vec();
        std::vector<float> log_value_vec(log_vec.size());
        TF_CHECK_OK(run_env->session->Run(inputs, log_vec, {}, &metrics));
        for(size_t j = 0; j < log_vec.size(); ++j) {
                log_value_vec[j] = *(metrics[j].scalar<float>().data());
        }
	gettimeofday(&end, 0);
        float millsecond = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
        log_value_vec.push_back(millsecond);
	return std::move(log_value_vec);
}

template<typename T1, typename T2> vector<float> Trainer<T1, T2>::train_online_base_file(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data)
{
	struct timeval begin, end;
	gettimeofday(&begin, 0);

	// fill input
	tensor_vector inputs;
	TF_CHECK_OK(run_env->session->Run(inputs, {}, {"ws_init"}, nullptr));
	MakeIOTensor<T1, T2> data_set2tensor(batch_x_data, mc->get_input_size(), batch_y_data, mc->get_output_size());
	inputs = {{"input_x", data_set2tensor.x}, {"input_y", data_set2tensor.y}};
	
	// pull parameters
	std::vector<tensorflow::Tensor> wb_tensor;
	param_handler->pull(wb_tensor);
	vector<float> weights_flatten;
        vector<size_t> weight_flatten_split{0};
	size_t cnt = 0;
	const std::vector<std::string>& dense_name_vec = mc->get_dense_name_vec();
	for(const auto tensor : wb_tensor) {
                inputs.push_back({dense_name_vec[cnt], tensor});
		weights_flatten.insert(weights_flatten.end(), tensor.flat<float>().data(), tensor.flat<float>().data() + tensor.NumElements());
		weight_flatten_split.push_back(weight_flatten_split[cnt] + tensor.NumElements());
		cnt++;
        }

	// forward/backword gradient
	const vector<string>& grad_op_vec = mc->get_gradient_name_vec();
	std::vector<tensorflow::Tensor> gradient_outputs;
	TF_CHECK_OK(run_env->session->Run(inputs, grad_op_vec, {}, &gradient_outputs));
	// parse gradient
	vector<float> grad_value_vec;
	for(size_t j = 0; j < grad_op_vec.size(); ++j) {
		const float* src_ptr = gradient_outputs[j].flat<float>().data();
		const vector<float> logits(src_ptr, src_ptr + gradient_outputs[j].NumElements());
		grad_value_vec.insert(grad_value_vec.end(), logits.begin(), logits.end());
	}
	
	// update parameter
	update(weights_flatten, grad_value_vec, 0.05);

	// push_back new weight into inputs
	const std::vector<std::vector<size_t>>& dense_shape = mc->get_dense_shape();
	inputs.erase(inputs.end() - dense_name_vec.size(), inputs.end());
	std::vector<tensorflow::Tensor> param_outputs;
	for(size_t i = 0; i < weight_flatten_split.size() - 1; i++) {
		auto tensor = make_tensor<float>(vector<float>(weights_flatten.begin() + weight_flatten_split[i], weights_flatten.begin() + weight_flatten_split[i + 1]), dense_shape[i]);
		inputs.push_back({dense_name_vec[i], tensor});
		param_outputs.push_back(tensor);
	}
	// run observe metrics
	std::vector<tensorflow::Tensor> metrics;
	const vector<string>& log_vec = mc->get_observe_name_vec();
	vector<float> log_value_vec(log_vec.size());
	TF_CHECK_OK(run_env->session->Run(inputs, log_vec, {}, &metrics));
	for(size_t j = 0; j < log_vec.size(); ++j) {
		log_value_vec[j] = *(metrics[j].scalar<float>().data());
	}
	// push weight
	param_handler->push(param_outputs);
	gettimeofday(&end, 0);
        float millsecond = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
	log_value_vec.push_back(millsecond);
	return log_value_vec;
}

template<typename T1, typename T2> vector<float> Trainer<T1, T2>::forward_propagate(const vector<T1> &input_data, const size_t sample_size) {
	vector<size_t> input_shape{sample_size};
	auto& origin_input_shape = mc->get_input_size();
	input_shape.insert(input_shape.end(), origin_input_shape.begin() + 1, origin_input_shape.end());
	OUTLOG(TS2(input_shape.size())) 
	Tensor input_tensor = make_tensor<T1>(input_data, input_shape);

	std::vector<tensorflow::Tensor> wb_tensor;
	param_handler->pull(wb_tensor);
	tensor_vector inputs = {{"input_x", input_tensor}};
	for(size_t i = 0; i < wb_tensor.size(); ++i) {
		inputs.push_back({mc->get_dense_name_vec()[i], wb_tensor.at(i)});
	}
	vector<tensorflow::Tensor> result_tensor;
	TF_CHECK_OK(run_env->session->Run(inputs, {"logits"}, {}, &result_tensor));
	const float* src_ptr = result_tensor[0].flat<float>().data();
	vector<float> logits(src_ptr, src_ptr+result_tensor[0].NumElements());
	return std::move(logits);
}

template<typename T1, typename T2> vector<int> Trainer<T1, T2>::predict(vector<T1> &input_data, size_t sample_size) {
	vector<float> logits = forward_propagate(input_data, sample_size);
	vector<int> pre_result;
	for(size_t b = 0; b < sample_size; ++b) {
		float max_value = FLT_MIN;
		int result_ind = -1;
		for (size_t i = b * mc->get_class_num(); i < mc->get_class_num() * (b + 1); ++i) { 
			if (logits[i] > max_value) {
				max_value = logits[i];
				result_ind = i - (b * mc->get_class_num());
			}
		}
		pre_result.push_back(result_ind);
	}
	stringstream ss;
	ss << "predict result:{";
	for_each(pre_result.begin(), pre_result.end(), [&ss](int val){ss << val << ",";});
	ss.seekp(-1, std::ios_base::end);
	ss << "}";
	cout << ss.str() << endl;
	return std::move(pre_result);
}

template<typename T1, typename T2> void Trainer<T1, T2>::update(vector<float> &weight, const vector<float> &grad, float learning_rate) {
	transform(weight.begin(), weight.end(), grad.begin(), weight.begin(), [learning_rate](const float base, const float grad) {return base - learning_rate * grad; });
}

template<typename T1, typename T2> void Trainer<T1, T2>::init_weight() {
	cout << "initializer weight..." << endl;
	param_handler->init_parameters("uniform", {-0.1, 0.1});
}

template<typename T1, typename T2> void Trainer<T1, T2>::train_offline(const dataset* ds, const int EPOCH)
{
	struct timeval begin, end;
	gettimeofday(&begin, 0);
	tensor_vector inputs;
	// init global variables
	TF_CHECK_OK(run_env->session->Run(inputs, {}, {"ws_init"}, nullptr)); // or TF_CHECK_OK(run_env.session->Run(inputs, {}, {"init"}, nullptr)); // "init" is default op?
	
	// run `train_step` to train
	std::vector<tensorflow::Tensor> metrics;
	MakeIOTensor<float, float> *data_set_tensor;
	const vector<string>& observe_vec = mc->get_observe_name_vec();
	for (size_t i = 0; i < EPOCH; ++i) {
		// Setup inputs and outputs:
		auto& [batch_x, batch_y] = ds->next_batch(mc->get_batch_size());
		data_set_tensor = new MakeIOTensor<float, float>(batch_x, mc->get_input_size(), batch_y, mc->get_output_size());
		inputs = {
			{"input_x", data_set_tensor->x},
			{"input_y", data_set_tensor->y}
		};
		TF_CHECK_OK(run_env->session->Run(inputs, observe_vec, {"train_step"}, &metrics));
		cout << "step:" << i;
		for(size_t j = 0; j < observe_vec.size(); ++j) {
			cout << "," << observe_vec[j] << ":" << *(metrics[j].scalar<float>().data());
		}
		cout << endl;
		delete data_set_tensor;
	}
	data_set_tensor = nullptr;

	std::vector<tensorflow::Tensor> param_outputs;
	TF_CHECK_OK(run_env->session->Run(inputs, mc->get_dense_name_vec(), {}, &param_outputs));
	param_handler->push(param_outputs);
	gettimeofday(&end, 0);
	float second = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
	cout << "time cost:" << second << "s" << endl;
}
}
