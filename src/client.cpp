#include <iostream>
#include <variant>
#include <vector>
#include <boost/program_options.hpp>  
#include "/usr/local/include/grpcpp/grpcpp.h"
#include "../proto/train_serving.grpc.pb.h"
#include "../utils/mnist_dataset.hpp"
#include "../utils/letters_dataset.hpp"
#include "../utils/criteo_dataset.hpp"
#include "../utils/common.h"
#include "model_conf.hpp"
#include "model_manager.h"

#define MAX_LEN 200*1024*1024
const size_t batch_size = 1024;
const string path = "../data/image_";
std::string grpc_addreass;

namespace bpo = boost::program_options;
ModelManager& model_manager = ModelManager::getInstance();

std::vector<float> get_images(const std::string al, int& size) {
	std::vector<std::string> digits = stringSplit(al, ',');
	size = digits.size();
        std::vector<float> data;
	for(auto digit : digits) {
        	auto temp = read_single_img(path + digit)();
        	data.insert(data.end(), temp.begin(), temp.end());
	}
	return std::move(data);
}

#define GET_CRITEO_DATASET(X) 	\
	X(lr)		        \
        X(fm)           	\
        X(ffm)          	\
        X(afm)          	\
        X(nfm)          	\
        X(widedeep)

shared_ptr<dataset> make_dataset(const string& model_name) {
#define MODEL_USE_DATASET(MODEL_NAME, DATASET) using MODEL_NAME = DATASET;
	MODEL_USE_DATASET(mlp, 		mnist_dataset)
	MODEL_USE_DATASET(lenet, 	mnist_dataset)
	MODEL_USE_DATASET(vgg, 		letters_dataset)
#define MODEL_USE_CRITEO(MODEL_NAME) MODEL_USE_DATASET(MODEL_NAME, criteo_dataset)
	GET_CRITEO_DATASET(MODEL_USE_CRITEO)
#define GET_MODEL_DATASET(DATASET) if (model_name == #DATASET) return make_shared<DATASET>();
	GET_MODEL_DATASET(mlp)
	GET_MODEL_DATASET(vgg)
	GET_MODEL_DATASET(lenet)
	GET_CRITEO_DATASET(GET_MODEL_DATASET)
}


class ClientAgent {
public:
	ClientAgent(const string& model_name) : model_name(model_name) {
    		// Set the maximum number of received and sent bytes.
    		channel_arg.SetMaxSendMessageSize(MAX_LEN);
		// Create a channel to connect to the server (with parameters).
		channel = grpc::CreateCustomChannel(grpc_addreass, grpc::InsecureChannelCredentials(), channel_arg);
	}

	void run_test_command_input_image_predict() {
		wt::PredictRequest req;
	        wt::PredictReply rsp;
		std::cout << "enter digit(separate with commas):";
		std::string digit;
		std::cin >> digit;
		int size = 0;
		std::vector<float> feature = get_images(digit, size);
		req.mutable_feature()->CopyFrom({feature.begin(), feature.end()});
		req.set_sample_size(size);
		req.set_model_name(model_name);
		std::unique_ptr<wt::Caca::Stub> stub = wt::Caca::NewStub(channel);
		grpc::Status status = stub->score(new grpc::ClientContext(), req, &rsp);
		if(status.ok()) {
			for(size_t i = 0; i < rsp.result_size(); ++i) {
				OUTLOG("answer:" + std::to_string(rsp.result(i)))
			}
		} else {
			OUTERR("predict error code:" + TS(static_cast<int>(status.error_code())) + "," + status.error_message())
		}
	}

	void run_online_train(int train_epoch) {
		wt::TrainRequest req;
		wt::TrainReply rsp;
		req.set_batch_size(batch_size);
		req.set_model_name(model_name);
 		shared_ptr<dataset> ds = make_dataset(model_name);
		const std::string model_json_path = MODEL_CONF_PATH + model_name + DOT_JSON;
		ModelConf mc(model_json_path);
		const std::vector<std::string>& metrics = mc.get_observe_name_vec();
		for(size_t i = 0; i < train_epoch; ++i) {
			auto&& [batch_x, batch_y] = ds->next_batch(batch_size);
			req.mutable_x()->CopyFrom({batch_x.begin(), batch_x.end()});
			req.mutable_y()->CopyFrom({batch_y.begin(), batch_y.end()});
			std::unique_ptr<wt::Caca::Stub> stub = wt::Caca::NewStub(channel);
			grpc::Status status = stub->training(new grpc::ClientContext(), req, &rsp);
			if(status.ok()) {
				std::stringstream ss;
				for (size_t i = 0; i < metrics.size(); ++i) {
					ss << metrics.at(i) << ":" + TS(rsp.result(i)) + ",";
				}
				ss << "time_cost:" << TS(rsp.result(i + 1)) << "ms.";
				OUTLOG(ss.str())
			} else {
				OUTERR("train error code:" + TS(static_cast<int>(status.error_code())) +  "," + status.error_message())
			}
			req.clear_x();
			req.clear_y();
		}
	}
private:
	grpc::ChannelArguments channel_arg;
	std::shared_ptr<grpc::Channel> channel; 
	string model_name;
};

int main(int argc, char const *argv[]) {
	bpo::options_description opt("options");
	std::string model_name;
	std::string s_type;
	int train_epoch;
	opt.add_options()
                ("grpc_addreass,a",   	bpo::value<std::string>(&grpc_addreass)->default_value("localhost:33333"),	"grpc addreass")
		("model,m", 		bpo::value<std::string>(&model_name), 						"model name")
		("serving_type,t", 	bpo::value<std::string>(&s_type)->default_value("training"), 						"training or predict")
		("epoch,e", 		bpo::value<int>(&train_epoch)->default_value(1),				"epoch if training")
		("help,h", 		"eg: ./tol_client -m mlp -t training/predict");
	bpo::variables_map vm;
	try {
		bpo::store(parse_command_line(argc, argv, opt), vm);
	} catch (...) {
		OUTLOG("augument error")
		return 0;
	}
	bpo::notify(vm); 
	if (vm.count("help") or vm.size() != 4) {
		std::cout << opt << std::endl;
		return 0;
	} else {
		OUTLOG("grpc_addreass:" + grpc_addreass + ",model_name:" + model_name + ",serving_type:" << s_type);
	}
	
	model_manager.get_model_index(model_name);
	unique_ptr<ClientAgent> ca(new ClientAgent(model_name));
	if (s_type == "training") {
		ca->run_online_train(train_epoch);
	} else if (s_type == "predict") {
		ca->run_test_command_input_image_predict();
	} else {
		std::cout << opt << std::endl;
		return 0;
	}
        return 0;
}
