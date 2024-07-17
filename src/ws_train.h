#include <iostream>
#include <sstream>
#include "run_env.hpp"
#include "../utils/dataset.hpp"
#include "tensor.hpp"
#include "parameter_io.hpp"
#include "model_conf.hpp"
#include "float.h"
#include <map>
#include <sys/time.h>

using namespace std;
using namespace tensorflow;
using tensor_vector = std::vector<std::pair<string, tensorflow::Tensor>>;


class ws_train {
public:
        ws_train() = delete;
        ws_train(const string& model_conf_path) {
                mc = new model_conf(model_conf_path);
                cout << *mc;
                run_env = new Run_Env(mc->get_pb_path());
                param_io = new Parameter_IO(mc->get_dense_name_vec(), mc->get_dense_shape(), mc->get_save_path_f());
        }
        virtual ~ws_train() {
                run_env->session->Close();
                delete run_env;
                delete mc;
                delete param_io;
        }

	[[deprecated("`train_online` is more recommended")]] 
        void                    train_offline(const dataset* ds, const int EPOCH = 100);
        vector<float>           train_online(const vector<float>& batch_x_data, const vector<float>& batch_y_data);
        vector<float>           forward_propagate(const vector<float> &input_data, const size_t sample_size);
        virtual vector<int>     predict(vector<float> &input_data, size_t sample_size);

        virtual void     	init_weight();
        virtual void     	update(vector<float> &weight, const vector<float> &grad, const float learning_rate = 0.05);

	//[[deprecated("`train_online` is more recommended")]] 
        //virtual void            run_train_offline(size_t epoch = 100) = 0;
private:
        Run_Env *run_env;
        model_conf *mc;
        Parameter_IO *param_io;
};
