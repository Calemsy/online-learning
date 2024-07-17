#include "ws_train.h"

void ws_train::train_offline(const dataset* ds, const int EPOCH)
{
	struct timeval begin, end;
	gettimeofday(&begin, 0);
	tensor_vector inputs;
	// init global variables
	TF_CHECK_OK(run_env->session->Run(inputs, {}, {"ws_init"}, nullptr)); // or TF_CHECK_OK(run_env.session->Run(inputs, {}, {"init"}, nullptr)); // "init" is default op?
	
	// run `train_step` to train
	std::vector<tensorflow::Tensor> loss_outputs;
	make_io_tensor<float, float> *data_set_tensor;
	vector<string> log_vec = mc->get_observe_name_vec();
	for (size_t i = 0; i < EPOCH; ++i) {
		// Setup inputs and outputs:
		vector<float> batch_x, batch_y;
		ds->next_batch(batch_x, batch_y, mc->get_batch_size());
		data_set_tensor = new make_io_tensor<float, float>(batch_x, mc->get_input_size(), batch_y, mc->get_output_size());
		inputs = {
			{"input_x", data_set_tensor->x},
			{"input_y", data_set_tensor->y}
		};
		TF_CHECK_OK(run_env->session->Run(inputs, log_vec, {"train_step"}, &loss_outputs));
		cout << "step:" << i;
		for(size_t j = 0; j < log_vec.size(); ++j) {
			cout << "," << log_vec[j] << ":" << *(loss_outputs[j].scalar<float>().data());
		}
		cout << endl;
		delete data_set_tensor;
	}
	data_set_tensor = nullptr;

	std::vector<tensorflow::Tensor> param_outputs;
	TF_CHECK_OK(run_env->session->Run(inputs, mc->get_dense_name_vec(), {}, &param_outputs));
	param_io->push(param_outputs);
	gettimeofday(&end, 0);
	float second = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
	cout << "time cost:" << second << "s" << endl;
}

vector<float> ws_train::train_online(const vector<float>& batch_x_data, const vector<float>& batch_y_data)
{
	struct timeval begin, end;
	gettimeofday(&begin, 0);
	// fill input
	tensor_vector inputs;
	TF_CHECK_OK(run_env->session->Run(inputs, {}, {"ws_init"}, nullptr));
	make_io_tensor<float, float> data_set_tensor(batch_x_data, mc->get_input_size(), batch_y_data, mc->get_output_size());
	inputs = {
		{"input_x", data_set_tensor.x},
		{"input_y", data_set_tensor.y}
	};
	std::vector<tensorflow::Tensor> wb_tensor;
	param_io->pull(wb_tensor);
	vector<float> weights_flatten;
	size_t cnt = 0;
	for(const auto tensor : wb_tensor) {
                inputs.push_back({mc->get_dense_name_vec()[cnt++], tensor});
		weights_flatten.insert(weights_flatten.end(), tensor.flat<float>().data(), tensor.flat<float>().data() + tensor.NumElements());
        }

	vector<string> grad_op_vec = mc->get_gradient_name_vec();
	std::vector<tensorflow::Tensor> gradient_outputs;
	// run gradient
	TF_CHECK_OK(run_env->session->Run(inputs, grad_op_vec, {}, &gradient_outputs));

	// parse gradient
	vector<float> grad_value_vec;
        vector<size_t> weight_flatten_split{0};
	for(size_t j = 0; j < grad_op_vec.size(); ++j) {
		const float* src_ptr = gradient_outputs[j].flat<float>().data();
		vector<float> logits(src_ptr, src_ptr + gradient_outputs[j].NumElements());
		weight_flatten_split.push_back(weight_flatten_split[j] + gradient_outputs[j].NumElements());
		grad_value_vec.insert(grad_value_vec.end(), logits.begin(), logits.end());
	}
	
	// update parameter
	update(weights_flatten, grad_value_vec, 0.05);

	// push_back new weight into inputs
	for(size_t i = 0; i < cnt; ++i) inputs.pop_back();
	std::vector<tensorflow::Tensor> param_outputs;
	for(size_t i = 0; i < weight_flatten_split.size()-1; i++) {
		auto tensor = make_tensor<float>(vector<float>(weights_flatten.begin() + weight_flatten_split[i], weights_flatten.begin() + weight_flatten_split[i + 1]), mc->get_dense_shape()[i]);
		inputs.push_back({mc->get_dense_name_vec()[i], tensor});
		param_outputs.push_back(tensor);
	}
	// run observe metrics
	std::vector<tensorflow::Tensor> loss_outputs;
	vector<string> log_vec = mc->get_observe_name_vec();
	vector<float> log_value_vec(log_vec.size());
	TF_CHECK_OK(run_env->session->Run(inputs, log_vec, {}, &loss_outputs));
	for(size_t j = 0; j < log_vec.size(); ++j) {
		log_value_vec[j] = *(loss_outputs[j].scalar<float>().data());
	}
	// push weight
	param_io->push(param_outputs);
	gettimeofday(&end, 0);
        float millsecond = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
	log_value_vec.push_back(millsecond);
	return log_value_vec;
}

vector<float> ws_train::forward_propagate(const vector<float> &input_data, const size_t sample_size) {
	vector<size_t> input_shape{sample_size};
	auto origin_size_vec = mc->get_input_size();
	input_shape.insert(input_shape.end(), origin_size_vec.begin() + 1, origin_size_vec.end());
	cout << "forward_propagate:input_shape:"; for (size_t i = 0; i < input_shape.size(); i++) cout << input_shape[i] << ","; cout << "input feature size:" << input_data.size() << endl;
	Tensor input_tensor = make_tensor<float>(input_data, input_shape);

	std::vector<tensorflow::Tensor> wb_tensor;
	param_io->pull(wb_tensor);
	tensor_vector inputs = {{"input_x", input_tensor}};
	size_t cnt = 0;
	for(const auto tensor : wb_tensor) {
		inputs.push_back({mc->get_dense_name_vec()[cnt++], tensor});
	}
	vector<Tensor> result_tensor;
	TF_CHECK_OK(run_env->session->Run(inputs, {"logits"}, {}, &result_tensor));
	const float* src_ptr = result_tensor[0].flat<float>().data();
	vector<float> logits(src_ptr, src_ptr+result_tensor[0].NumElements());
	return std::move(logits);
}

vector<int> ws_train::predict(vector<float> &input_data, size_t sample_size) {
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

void ws_train::update(vector<float> &weight, const vector<float> &grad, float learning_rate) {
	size_t size = weight.size();
	for(size_t i = 0; i < size; ++i) weight[i] = weight[i] - learning_rate * grad[i];
}

void ws_train::init_weight() {
	cout << "initializer weight..." << endl;
	param_io->init("uniform", {-0.1, 0.1});
}
