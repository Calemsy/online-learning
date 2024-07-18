#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <fstream>
#include "../utils/common.h"

class ModelManager {
public:
	static ModelManager& getInstance() {
		static ModelManager manager;
		return manager;
	}
	inline uint32_t get_model_index(const std::string& model_name) const { 
		if (model2index_map.count(model_name) == 0) {
			OUTERR(model_name + " not in `model_list`")
			std::exit(EXIT_FAILURE);
		}
		return model2index_map.at(model_name);
	}
	/* Since the Key type of ps-lite does not support the string type (if it did support it, we could directly use the concatenation 
 	 * of model_name and dense_name as the key for push or pull), restrictions have been imposed on the range of keys used for each 
 	 * model here to prevent the confusion of parameters of different models. Specifically, the number of each model is specified in 
 	 * the file `model_list`. Here, the begining value of the key for a model is obtained through shift operations. For example, for
 	 * a model with a model number of 2, the range of its key is [2048, 3072), 2048=2<<10. A uint32_t (an unsigned 32-bit integer) 
 	 * type is used here as the value type for all Key, so the maximum number of supported models is 2^22 = 4194304, and the maximum
 	 * number of parameters supported for each model is 2^10 = 1024.
 	*/
	inline uint32_t get_model_pskey_offset(const std::string& model_name) const { return get_model_index(model_name) << 10;}
	ModelManager(const ModelManager&) = delete;
	ModelManager& operator=(const ModelManager&) = delete;
private:
	ModelManager() {
		const std::string file_name = "../model/model_list";
		std::ifstream file(file_name);
		if (!file.is_open()) {
        		OUTERR("model_list can't open")
			return;
    		}
		std::string line;
		while (std::getline(file, line)) {
			std::istringstream iss(line);
			std::string key, val;
			if (iss >> key >> val) {
				model2index_map[key] = std::stoi(val);
			} else {
            			OUTERR("parse error: ../model/model_list")
        		}
		}
		file.close();
		for(const auto& iter : model2index_map) {
			OUTLOG("model name=" + iter.first + ", model index:" + TS(iter.second))
		}
	}
	std::unordered_map<std::string, uint32_t> model2index_map;
};
