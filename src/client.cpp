#include <iostream>
#include <vector>
#include <boost/program_options.hpp>  
#include "/usr/local/include/grpcpp/grpcpp.h"
#include "../proto/train_serving.grpc.pb.h"
#include "../utils/mnist_dataset.hpp"
#include "../utils/letters_dataset.hpp"
#include "../utils/common.h"

#define MAX_LEN 200*1024*1024
const size_t batch_size = 1024;

namespace bpo = boost::program_options;

std::vector<float> get_images(const std::string al, int& size) {
        const string path = "/data0/users/shuaishuai3/wt/t/t1_8/data/image_";
	std::vector<std::string> digits = stringSplit(al, ',');
	size = digits.size();
        std::vector<float> data;
	for(auto digit : digits) {
        	auto temp = read_single_img(path + digit)();
        	data.insert(data.end(), temp.begin(), temp.end());
	}
	return std::move(data);
}

shared_ptr<dataset> make_dataset(const string& model_name) {
#define MODEL_USE_DATASET(MODEL_NAME, DATASET) using MODEL_NAME = DATASET;
	MODEL_USE_DATASET(mlp_mnist, 	mnist_dataset)
	MODEL_USE_DATASET(lenet_mnist, 	mnist_dataset)
	MODEL_USE_DATASET(vgg_letters, 	letters_dataset)
#define GET_MODEL_DATASET(MODEL_NAME, DATASET) if (MODEL_NAME == #DATASET) return make_shared<DATASET>();
	GET_MODEL_DATASET(model_name, mlp_mnist)
	GET_MODEL_DATASET(model_name, lenet_mnist)
	GET_MODEL_DATASET(model_name, vgg_letters)
}

class Client_Agent {
public:
	Client_Agent(const string& model_name) : model_name(model_name) {
    		// 设置最大接收和发送字节数
    		channel_arg.SetMaxSendMessageSize(MAX_LEN);
		// 创建一个连接服务器的通道(带参数)
		channel = grpc::CreateCustomChannel("localhost:33333", grpc::InsecureChannelCredentials(), channel_arg);
		model_json_conf_path = path_prefix + model_name + path_suffix;
	}

	void predict() {
		wt::PredictRequest req;
	        wt::PredictReply rsp;
		std::cout << "enter digit(separate with commas):";
		std::string digit;
		std::cin >> digit;
		int size = 0;
		std::vector<float> feature = get_images(digit, size);
		req.mutable_feature()->CopyFrom({feature.begin(), feature.end()});
		req.set_sample_size(size);
		req.set_model_name(model_json_conf_path);
		std::unique_ptr<wt::Caca::Stub> stub = wt::Caca::NewStub(channel);
		grpc::Status status = stub->score(new grpc::ClientContext(), req, &rsp);
		if(status.ok()) {
			for(size_t i = 0; i < rsp.result_size(); ++i)
				std::cout << rsp.result(i) << std::endl;
		} else {
			cout << "predict error code:" << status.error_code() << "," << status.error_message() << endl;
		}
	}

	void train_online(int train_epoch) {
		wt::TrainRequest req;
		wt::TrainReply rsp;
		vector<float> batch_x, batch_y;
 		shared_ptr<dataset> ds = make_dataset(model_name);
		for(size_t i = 0; i < train_epoch; ++i) {
			ds->next_batch(batch_x, batch_y, batch_size);
			req.mutable_x()->CopyFrom({batch_x.begin(), batch_x.end()});
			req.mutable_y()->CopyFrom({batch_y.begin(), batch_y.end()});
			req.set_batch_size(batch_size);
			req.set_model_name(model_json_conf_path);
			std::unique_ptr<wt::Caca::Stub> stub = wt::Caca::NewStub(channel);
			grpc::Status status = stub->training(new grpc::ClientContext(), req, &rsp);
			if(status.ok()) {
				cout << "loss:" << rsp.result(0) << ",acc:" << rsp.result(1) << std::endl;
			} else {
				cout << "train error code:" << status.error_code() << "," << status.error_message() << endl;
			}
			batch_x.clear();
			batch_y.clear();
			req.clear_x();
			req.clear_y();
		}
	}
private:
	grpc::ChannelArguments channel_arg;
	// 创建一个连接服务器的通道
	//channel_arg.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, MAX_LEN);
	//std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel("localhost:33333", grpc::InsecureChannelCredentials());
	std::shared_ptr<grpc::Channel> channel; 
	string model_name;
	string model_json_conf_path;
	const string path_prefix = "/data0/users/shuaishuai3/wt/t/t1_8/model/model_conf/";
	const string path_suffix = ".json";
};

int main(int argc, char const *argv[]) {
	bpo::options_description opt("options");
	static std::string model_name;
	std::string s_type;
	int train_epoch;
	opt.add_options()
		("model,m", 		bpo::value<std::string>(&model_name), 		"model name")
		("serving_type,t", 	bpo::value<std::string>(&s_type), 		"online training or predict check")
		("epoch,e", 		bpo::value<int>(&train_epoch)->default_value(1),"epoch if training")
		("help,h", 		"eg: ./client -m mlp_mnist -t training/predict");
	bpo::variables_map vm;
	try {
		bpo::store(parse_command_line(argc, argv, opt), vm);
	} catch (...) {
		cout << "augument error" << endl;
		return 0;
	}
	bpo::notify(vm); 
	if (vm.count("help") or vm.size() == 1) {
		std::cout << opt << std::endl;
		return 0;
	} else {
		cout << "model_name:" << model_name << ",serving_type:" << s_type << endl;
	}
	
	unique_ptr<Client_Agent> ca(new Client_Agent(model_name));
	if (s_type == "training") {
		ca->train_online(train_epoch);
	} else if (s_type == "predict") {
		ca->predict();
	} else {
		std::cout << opt << std::endl;
		return 0;
	}
        return 0;
}
