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

namespace tol{
template<typename T1, typename T2>
class Trainer {
public:
	/**
	 * @brief Deleted default constructor.
	 *
	 * This constructor is explicitly deleted to prevent the creation of a `Trainer` object without providing the necessary parameters.
	 * It ensures that the `Trainer` class can only be instantiated using the parameterized constructor.
	 */
        Trainer() = delete;

	/**
	 * @brief Constructor for the Trainer class.
	 *
	 * This constructor initializes a new instance of the Trainer class with the given model configuration path,
	 * parameter backend type, worker ID of ps, key offset for store, and total number of PS workers. It sets up
	 * the necessary components for training, including loading the model configuration, initializing the run 
	 * environment, and creating the appropriate parameter I/O handler based on the specified backend.
	 *
	 * @param model_conf_path The path to the model configuration file.
	 * @param backend The type of parameter backend to use (e.g., IO or PS, recommend: IO is used for testing, 
	 * while PS is used for the production environment).
	 * @param worker_id The ID of the ps-worker. If the backend selected is "IO", then all the parameters starting
	 * from here can be ignored. All the remaining parameters are related to backend is "PS".
	 * @param ps_key_offset The offset of PS-keys. The offset of key is used to distinguish the parameters of 
	 * different models for save in ps-server. For example, the key of Model1 starts from 0, and the key of Model2 
	 * starts from 1024.
	 * @param total_ps_worker The total number of PS workers. The quantity here needs to be in line with the 
	 * parameters specified when starting the PS server
	 */
        Trainer(const string& model_conf_path, const Parameter_backend backend, int worker_id = 0, uint32_t ps_key_offset = 0, int total_ps_worker = 8) : 
		ps_backend(backend),
		worker_id(worker_id),
		ps_key_offset(ps_key_offset),
		total_ps_worker(total_ps_worker)
	{
                mc = new ModelConf(model_conf_path);
                cout << *mc;
                run_env = new TFRunEnv(mc->get_pb_path());
		if (ps_backend == IO) {
                	param_handler = new ParameterBaseFile(mc->get_model_name(), mc->get_dense_name_vec(), mc->get_dense_shape(), 
						    mc->get_save_path_f());
		} else {
			OUTLOG("worker_id:" + TS(worker_id))
			param_handler = new ParameterBasePS(mc->get_model_name(), mc->get_dense_name_vec(), mc->get_dense_shape(), 
						    worker_id, ps_key_offset, total_ps_worker);
		}
        }

        virtual ~Trainer() {
		run_env->session->Close();
                delete run_env;
                delete mc;
                delete param_handler;
		mc = nullptr;
		param_handler = nullptr;
		run_env = nullptr;
        }

	/**
	 * @brief Trains the model using the offline training method.
	 *
	 * This method trains the model on a given dataset for a specified number of epochs.
	 *
	 * @param ds A pointer to the dataset.
	 * @param EPOCH The number of epochs to train the model. Default is 100.
	 */
	[[deprecated("`train_online` is more recommended")]] 
        void                    train_offline(const dataset* ds, const int EPOCH = 100);

	/**
	 * @brief Trains the model using the online training method.
	 *
	 * This method trains the model on a given batch of data. It takes two vectors as input:
	 * one for the input features (`batch_x_data`) and one for the target labels (`batch_y_data`).
	 * The method returns a vector of float representing the loss or other metrics after training on the batch.
	 * This method is the core of this project.
	 *
	 * @param batch_x_data A vector containing the input features for the batch.
	 * @param batch_y_data A vector containing the target labels for the batch.
	 * @return A vector of floats representing the loss or other metrics after training on the batch.
	 */
        vector<float>           train_online(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data);

	/**
	 * @brief Performs forward propagation through the model.
	 *
	 * This method takes input data and performs a forward pass through the model to produce the output predictions.
	 * It is typically used during inference or evaluation of the model.
	 *
	 * @param input_data A vector containing the input features for the samples.
	 * @param sample_size The number of samples in the input data.
	 * @return A vector of floats representing the output predictions from the model.
	 */
        vector<float>           forward_propagate(const vector<T1> &input_data, const size_t sample_size);

        virtual vector<int>     predict(vector<T1> &input_data, size_t sample_size);
        virtual void     	init_weight();
        virtual void     	update(vector<float> &weight, const vector<float> &grad, const float learning_rate = 0.05);

	//[[deprecated("`train_online` is more recommended")]] 
        //virtual void            run_train_offline(size_t epoch = 100) = 0;
private:
        vector<float>           train_online_base_ps(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data);
        vector<float>           train_online_base_file(const vector<T1>& batch_x_data, const vector<T2>& batch_y_data);
        TFRunEnv*		run_env;
        ModelConf*		mc;
        ParameterPP*		param_handler;
	const Parameter_backend	ps_backend;
	// ps conf
	int 			worker_id;
	uint32_t 		ps_key_offset;
	uint8_t			total_ps_worker;
};
}
