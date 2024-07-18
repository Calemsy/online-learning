#include "ps/ps.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <numeric>
#include <iterator>

using namespace std;
using namespace ps;

void RunWorker(int customer_id) {
  	cout << "Start Worker rank=" << MyRank() << "\n";
  	KVWorker<float> kv(0, 0);

  	// init
  	int num = 5;
  	std::vector<Key> keys(num);
	std::vector<int> lens{3,4,9,2,1};
  	std::vector<float> vals(0);

  	for (int i = 0; i < num; ++i) {
    		keys[i] = i;
		vector<float> val(lens[i], i * 1.0f);
    		std::copy(val.begin(), val.end(), std::back_inserter(vals));
  	}

  	// push
  	int repeat = 1;
  	std::vector<int> ts;
  	for (int i = 0; i < repeat; ++i) {
    		ts.push_back(kv.Push(keys, vals, lens));
  	}
  	for (int t : ts) kv.Wait(t);

  	// pull
  	std::vector<float> rets;
  	kv.Wait(kv.Pull(keys, &rets));
	std::vector<int> split_vec(lens.size() + 1, 0);
	for(size_t i = 1; i < split_vec.size(); ++i) {
		split_vec.at(i) = split_vec.at(i - 1) + lens.at(i - 1);
	}
	for(size_t i = 0; i < split_vec.size() - 1; ++i) {
		size_t start = split_vec.at(i), end = split_vec.at(i + 1);
		cout << "key=" << keys.at(i) << " ";
		for(auto iter = rets.begin() + start; iter != rets.begin() + end; ++iter) {
			cout << *iter << ", ";
		}
		cout << endl;
	}
  	cout << "\nFinal\n";
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		std::cout << "./ps_client 0/1 .." << std::endl;
		return -1;
	}
	
	setenv("DMLC_NUM_SERVER", "1", 1); 
    	setenv("DMLC_NUM_WORKER", "2", 1);
	setenv("DMLC_PS_ROOT_URI", "127.0.0.1", 1);
	setenv("DMLC_PS_ROOT_PORT", "8111", 1);
	setenv("DMLC_ROLE", "worker", 1);
	int customer_id = std::stoi(argv[1]);
	char work_id[] = "./W";
	strcat(work_id, argv[1]);
	setenv("HEAPPROFILE", work_id, 1); 
	
	cout << "before Start()" << endl;
	Start(0);
	cout << "Start.." << endl;
  	RunWorker(customer_id);
  	Finalize(0, true);
  	return 0;
}
