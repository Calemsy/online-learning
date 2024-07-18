#include <iostream>
#include <sstream>
#include <map>
#include <sys/time.h>
#include "run_env.hpp"
#include "tensor.hpp"
#include "parameter_io.hpp"
#include "parameter_ps.hpp"
#include "model_conf.hpp"
#include "float.h"
#include "../utils/dataset.hpp"
#include "../utils/common.h"

using namespace std;
using namespace tensorflow;
using tensor_vector = std::vector<std::pair<string, tensorflow::Tensor>>;


template<typename T1, typename T2>
class ws_train {
public:
        ws_train() = delete;
        ws_train(const string& model_conf_path, Parameter_backend backend, int worker_id = 0, uint32_t ps_key_offset = 0, int total_ps_worker = 8) : 
		ps_backend(backend),
		worker_id(worker_id),
		ps_key_offset(ps_key_offset),
		total_ps_worker(total_ps_worker)
	{
                mc = new model_conf(model_conf_path);
                cout << *mc;
                run_env = new Run_Env(mc->get_pb_path());
		if (ps_backend == IO) {
                	param_io = new Parameter_IO(mc->get_model_name(), mc->get_dense_name_vec(), mc->get_dense_shape(), 
						    mc->get_save_path_f());
		} else {
			OUTLOG("worker_id:" + std::to_string(worker_id))
			param_io = new Parameter_Ps(mc->get_model_name(), mc->get_dense_name_vec(), mc->get_dense_shape(), 
						    worker_id, ps_key_offset, total_ps_worker);
		}
        }
        virtual ~ws_train() {
		//tensorflow::Status status = run_env->session->Close();
		run_env->session->Close();
                delete run_env;
                delete mc;
                delete param_io;
		mc = nullptr;
		param_io = nullptr;
		run_env = nullptr;
        }

	[[deprecated("`train_online` is more recommended")]] 
        void                    train_offline(const dataset<T1, T2>* ds, const int EPOCH = 100);
        vector<float>           train_online(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data);
        vector<float>           forward_propagate(const vector<T1> &input_data, const size_t sample_size);
        virtual vector<int>     predict(vector<T1> &input_data, size_t sample_size);

        virtual void     	init_weight();
        virtual void     	update(vector<float> &weight, const vector<float> &grad, const float learning_rate = 0.05);

	//[[deprecated("`train_online` is more recommended")]] 
        //virtual void            run_train_offline(size_t epoch = 100) = 0;
private:
        Run_Env*		run_env;
        model_conf*		mc;
        Parameter_PAP*		param_io; // TODO change name
	const Parameter_backend	ps_backend;
	// ps conf
	int 			worker_id;
	uint32_t 		ps_key_offset;
	uint8_t			total_ps_worker;

private:
        vector<float>           train_online_base_ps(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data);
        vector<float>           train_online_base_file(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data);
};
