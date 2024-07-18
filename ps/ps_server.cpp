#include <iostream>
#include <signal.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include "ps/ps.h"
#include "../utils/common.h"

using namespace ps;
namespace bpo = boost::program_options;

volatile sig_atomic_t terminate = 0;
std::unordered_map<Key, std::vector<float>> store;
std::string role;

template <typename Val> struct KVServerHandle {
        void operator()(const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
                size_t n = req_data.keys.size(); 
                KVPairs<Val> res; 
                if (req_meta.push) {
                        CHECK_EQ(n, req_data.lens.size());
                } else {
                        res.keys = req_data.keys;
                }
		auto lens = req_data.lens;
		std::vector<int> split_vec(lens.size() + 1, 0);
		std::partial_sum(lens.begin(), lens.end(), split_vec.begin() + 1);
                for (size_t i = 0; i < n; ++i) {
                        Key key = req_data.keys[i];
                        if (req_meta.push) {
				int start = split_vec.at(i), stop = split_vec.at(i + 1);
                                if (store.count(key) == 0) {
					store[key].assign(req_data.vals.begin() + start, req_data.vals.begin() + stop);
					OUTLOG("init push key=" + TS(key) + ",init size:" + TS(store.at(key).size()))
				} else {
					OUTLOG("push key=" + TS(key) + ",val.size:" + TS(store.at(key).size()) + ",[" + TS(start) + "," + TS(stop) + "]")
                    			transform(store.at(key).begin(), store.at(key).end(), req_data.vals.begin() + start, store.at(key).begin(), [](const float base, const float grad){return base - 0.05 * grad;});            
				}
                        } else { // is pull
				if (store.count(key)) {
					SArray<Val> val(store.at(key));
					res.vals.append(val);
					OUTLOG("pull key=" + TS(key) + ",val.size:" + TS(val.size()) + ",accumulate.size:" + TS(res.vals.size()))
				}
                        }
                }
                server->Response(req_meta, res);
        }
};

void StartServer() {
        if (!IsServer()) return;
        OUTLOG("Start Server...")
        auto server = new KVServer<float>(0);
        server->set_request_handle(KVServerHandle<float>());
        RegisterExitCallback([server](){ delete server; });
        OUTLOG("Server End.")
}

void signal_handler(int signum) {
	const std::string message = (role == "server") ? " to save." : (role + " exit.");
    	OUTLOG(std::string("received [") + strsignal(signum) + "], " + message)
	auto save2store = []() {
		std::fstream fout("../save/store.dt", std::fstream::out | std::fstream::binary);
		if(!fout) throw std::runtime_error("save: open store file error");
		for(auto iter = store.begin(); iter != store.end(); ++iter) {
			OUTLOG("save key=" + TS(iter->first) + ",size=" + TS(iter->second.size()))
			size_t val_size = iter->second.size();
			fout.write((char*)&(iter->first), sizeof(Key));
			fout.write((char*)&val_size, sizeof(size_t));
			fout.write((char*)iter->second.data(), val_size * sizeof(float));
		}
		fout.close();
	};
	if (role == "server") {
		save2store();
		OUTLOG("save `store` end")
	}
    	terminate = 1;
}

void load_store() {
	std::fstream fin("../save/store.dt", std::fstream::in | std::fstream::binary);
	if(!fin) {
		OUTLOG("load: open store file error")
		return ;
	}
	while (!fin.eof()) {
		char ch = fin.peek();
                if (ch == -1) break;
		Key key;
                fin.read((char*)&key, sizeof(Key));
                size_t param_size;
                fin.read((char*)&param_size, sizeof(size_t));
                OUTLOG("load key:" + TS(key) + ",size:" + TS(param_size));
                std::vector<float> params(param_size);
                fin.read((char*)params.data(), sizeof(float) * param_size);
                store.insert({key, params});
        }
	fin.close();
}

int main(int argc, char* argv[]) {
	bpo::options_description opt("options");
	int worker_num;
	int server_num;
	opt.add_options()
		("role,r", 	bpo::value<std::string>(&role), 		"scheduler or server in ps-lite")
		("worker_num,w",bpo::value<int>(&worker_num)->default_value(1),	"worker numbers")
		("server_num,s",bpo::value<int>(&server_num)->default_value(1), "server numbers")
		("help,h", "eg: ./ps_server -r scheduler/server -s 1 -w 2 Note: first start scheduler");
	bpo::variables_map vm;
	try {
                bpo::store(parse_command_line(argc, argv, opt), vm);
        } catch (...) {
                OUTLOG("augument error")
                return 0;
        }
        bpo::notify(vm);
        if (vm.count("help") or vm.size() != 3) {
                std::cout << opt << std::endl;
                return 0;
        } else {
		OUTLOG("role:" + role + ",work num:" + TS(worker_num) + ",server num:" + TS(server_num))
	}
	
	struct sigaction sa;
	sa.sa_handler = signal_handler;
	sa.sa_flags = 0; 
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);

	const std::unordered_map<std::string, std::string> PS_ENV = {
		{"DMLC_NUM_SERVER", std::to_string(server_num)},
		{"DMLC_NUM_WORKER", std::to_string(worker_num)},
		{"DMLC_PS_ROOT_URI", "127.0.0.1"},
		{"DMLC_PS_ROOT_PORT", "8111"},
	};
	for(const auto& config : PS_ENV) {
		setenv(config.first.c_str(), config.second.c_str(), 1);
	}

	if (role == "server") {
		OUTLOG("is server")
        	setenv("DMLC_ROLE", "server", 1);
        	setenv("HEAPPROFILE", (std::string("./S") + std::to_string(server_num - 1)).c_str(), 1);
		load_store();
		OUTLOG("load into store done.")
	} else if (role == "scheduler") {
		OUTLOG("is scheduler")
        	setenv("DMLC_ROLE", "scheduler", 1);
	} else {
		OUTLOG("role is error")
		return -1;
	}

	OUTLOG("running ...")
	while (!terminate) {
		Start(0);
		StartServer();
		Finalize(0, true);
	}
	OUTLOG("bye")
    	return 0;
}
