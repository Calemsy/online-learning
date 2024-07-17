#include <memory>
#include "../proto/train_serving.grpc.pb.h"
#include <iostream>
#include "/usr/local/include/grpcpp/grpcpp.h"
#include <vector>
#include <numeric>
#include "ws_train.cpp" // 这里为啥不能包含.h
#define MAX_LEN 200*1024*1024

/*
#define NEW_TRAINER(MODEL_NAME, TRAINER) 		\
if (MODEL_NAME == #TRAINER) 				\
return make_shared<TRAINER>(MODEL_NAME);

#define CONFIG_MODEL_TRAINER(MODEL_NAME, TRAINER)	\
using MODEL_NAME = TRAINER;

shared_ptr<ws_train> make_trainer(string model_name) {
	CONFIG_MODEL_TRAINER(mlp_mnist,		mnist_train)
	CONFIG_MODEL_TRAINER(lenet_mnist, 	mnist_train)
	NEW_TRAINER(model_name, 		mlp_mnist)
	NEW_TRAINER(model_name, 		lenet_mnist)
}
*/

class ServiceImpl: public wt::Caca::Service {
public:
	grpc::Status score(grpc::ServerContext* context, const wt::PredictRequest* request, wt::PredictReply* response) { 
		std::vector<float> feature;
		size_t sample_size = request->sample_size();
		feature.insert(feature.end(), request->feature().begin(), request->feature().end());
		// trainer = make_trainer(request->model_name());
		trainer = make_shared<ws_train>(request->model_name());
		auto result = trainer->predict(feature, sample_size);
		for(size_t i = 0; i < sample_size; ++i)
			response->add_result(result[i]);
		return grpc::Status::OK;
	}

	grpc::Status training(grpc::ServerContext* context, const wt::TrainRequest* request, wt::TrainReply* response) {
		std::vector<float> train_x, train_y;
		train_x.insert(train_x.end(), request->x().begin(), request->x().end());
		train_y.insert(train_y.end(), request->y().begin(), request->y().end());
		trainer = make_shared<ws_train>(request->model_name());
		cout << ",request_from:" << context->peer() << ",batch_size:" << request->batch_size() << endl;
		vector<float> result = trainer->train_online(train_x, train_y);
		for(size_t i = 0; i < result.size(); ++i) {
			response->add_result(result[i]);
		}
		return grpc::Status::OK;
	}
	~ServiceImpl() = default;
private:
	std::shared_ptr<ws_train> trainer;
};

int main()
{
	grpc::ServerBuilder builder;  // 服务构建器，用于构建同步/异步服务
	builder.SetMaxReceiveMessageSize(MAX_LEN);
	// SetMaxSendMessageSize
	builder.AddListeningPort("0.0.0.0:33333", grpc::InsecureServerCredentials());  // 添加监听的地址和端口，后一个参数用于设置认证方式，这里选择不认证
	ServiceImpl service;  // 创建服务对象
	builder.RegisterService(&service);  // 注册服务
	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());  // 构建服务器
	std::cout<<"server runing..."<<std::endl;
	server->Wait();  // 进入服务处理循环（必须在某处调用server->Shutdown()才会返回）
	return 0;
}
