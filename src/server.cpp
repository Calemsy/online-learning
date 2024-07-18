#include <memory>
#include <iostream>
#include <vector>
#include <numeric>
#include <boost/program_options.hpp>
#include <mutex>
#include "/usr/local/include/grpcpp/grpcpp.h"
#include "../proto/train_serving.grpc.pb.h"
#include "ol_trainer.cpp"
#include "model_manager.h"
#include "../utils/common.h"

#define MAX_LEN 	200*1024*1024
ModelManager& model_manager = ModelManager::getInstance();

namespace bpo = boost::program_options;
using namespace tol;
std::mutex mtx;

class ServiceImpl: public wt::Caca::Service {
public:
	ServiceImpl(const Parameter_backend ps_backend, const int total_works) 
		: ps_backend(ps_backend)
		, total_works(total_works) {
	}

	// TODO ps version
	grpc::Status score(grpc::ServerContext* context, const wt::PredictRequest* request, wt::PredictReply* response) { 
		size_t sample_size = request->sample_size();
		std::vector<float> feature;
		feature.insert(feature.end(), request->feature().begin(), request->feature().end());
		auto model_name = request->model_name();
		uint32_t ps_key_offset = model_manager.get_model_pskey_offset(model_name);
		std::string model_json_path = MODEL_CONF_PATH + model_name + DOT_JSON;
		auto* trainer = new Trainer<float, float>(model_json_path, ps_backend, worker_id, ps_key_offset); // TODO worker_id ...
		auto result = trainer->predict(feature, sample_size);
		for(size_t i = 0; i < sample_size; ++i) {
			response->add_result(result[i]);
		}
		delete trainer;
		trainer = nullptr;
		return grpc::Status::OK;
	}

	grpc::Status training(grpc::ServerContext* context, const wt::TrainRequest* request, wt::TrainReply* response) {
		std::vector<float> train_x, train_y;
		train_x.insert(train_x.end(), request->x().begin(), request->x().end());
		train_y.insert(train_y.end(), request->y().begin(), request->y().end());
		auto model_name = request->model_name();
		uint32_t ps_key_offset = model_manager.get_model_pskey_offset(model_name);
		std::string model_json_path = MODEL_CONF_PATH + model_name + DOT_JSON;
		auto wi = assign_worker_id();
		OUTLOG("running training: from:" + context->peer() + ",model:" + request->model_name() + ",batch_size:" + TS(request->batch_size()) + ",ps_key_offset:" + TS(ps_key_offset) + "," + TS2(wi));
		auto* trainer = new Trainer<float, float>(model_json_path, ps_backend, wi, ps_key_offset, total_works);
		vector<float> result = trainer->train_online(train_x, train_y);
		for(size_t i = 0; i < result.size(); ++i) {
			response->add_result(result[i]);
		}
		delete trainer;
		trainer = nullptr;
		return grpc::Status::OK;
	}
	~ServiceImpl() = default;
private:
	const Parameter_backend ps_backend;
	const int total_works;
	mutable int worker_id = 0;
	inline int assign_worker_id() const {
		std::unique_lock<std::mutex> lock(mtx);
		int val = worker_id++;
		if (worker_id == total_works) {
			worker_id = 0;
		}
		return val;
	}
};

int main(int argc, char const *argv[]) {
	bpo::options_description opt("options");
	std::string grpc_addreass;
        std::string ps_backend;
        int total_worker_num;
        opt.add_options()
                ("grpc_addreass,a",   bpo::value<std::string>(&grpc_addreass)->default_value("0.0.0.0:33333"),	"grpc addreass")
                ("ps_backend,p",      bpo::value<std::string>(&ps_backend),                			"ps backend, ps(recommend) or io")
                ("total_worker_num,n",bpo::value<int>(&total_worker_num)->default_value(8),			"ps total worker number")
                ("help,h",            "eg: ./tol_server -p ps -n 8");
        bpo::variables_map vm;
        try {
                bpo::store(parse_command_line(argc, argv, opt), vm);
        } catch (...) {
                OUTERR("augument error")
                return 0;
        }
        bpo::notify(vm);
        if (vm.count("help") or vm.size() != 3) {
                std::cout << opt << std::endl;
                return 0;
        } else {
                OUTLOG("grpc addreass:" + grpc_addreass + ",ps backend:" + ps_backend + ",ps worker num:" + TS(total_worker_num));
        }

	Parameter_backend ps_type = (Parameter_backend)(ps_backend == "ps" ? PS : IO);
	grpc::ServerBuilder builder;  
	builder.SetMaxReceiveMessageSize(MAX_LEN);
	builder.AddListeningPort(grpc_addreass, grpc::InsecureServerCredentials());  
	ServiceImpl service(ps_type, total_worker_num); 
	builder.RegisterService(&service);  
	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());  
	OUTLOG("grpc server runing ...")
	server->Wait();  
	return 0;
}
