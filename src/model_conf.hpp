#include <iostream>
#include "json.h"
#include <fstream>
#include <algorithm>
#include <vector>

#define PARSE_VALUE(member, type) member = root["other"][#member].as##type();

#define PARSE_LIST_VALUE(member, type)							\
for(unsigned int i = 0; i < root["other"][#member].size(); ++i) {			\
member##_vec.push_back(root["other"][#member][i].as##type());			        \
}

#define PRINT_V(v) cout << #v << ":[";for(size_t i = 0; i < v.size(); ++i) cout << v[i] << ",]"[i==v.size()-1];cout << ",";
#define PRINT_VV(vv) cout << #vv << ":[";for_each(vv.begin(), vv.end(), [](const auto& _){PRINT_V(_)});cout << "\b]";

class model_conf {
friend std::ostream& operator<<(std::ostream& out, const model_conf& model);
public:
	model_conf() = delete;
	model_conf(const std::string& json_path) : json_path(json_path) {
		read_from_file();
	}
	const std::vector<std::string>& get_dense_name_vec() const 	{return dense_vec;}
	const std::vector<std::string>& get_gradient_name_vec() const 	{return gradient_vec;}
	const std::vector<std::string>& get_observe_name_vec() const	{return observe_vec;}
	const std::vector<std::vector<size_t>>& get_dense_shape() const	{return dense_shape;}
	const std::vector<size_t> get_input_size() const 		{return input_size_vec;}
	const std::vector<size_t> get_output_size() const 		{return output_size_vec;}
	const size_t get_batch_size() const 				{return batch_size;}
	const size_t get_class_num() const 				{return class_num;}
	const std::string& get_model_name() const			{return model_name;}
	const std::string& get_pb_path() const 				{return pb_path;}
	const std::string& get_json_path() const 			{return json_path;}
	const std::string get_save_path_f() const			{return "../save/" + model_name;}
private:
	void read_from_file() {
		Json::Reader reader;
		Json::Value root;
		std::ifstream infile(json_path);
		if(reader.parse(infile, root)) {
			PARSE_LIST_VALUE(dense, String)
			PARSE_LIST_VALUE(gradient, String)
			PARSE_LIST_VALUE(observe, String)
			PARSE_LIST_VALUE(input_size, Int)
			PARSE_LIST_VALUE(output_size, Int)
			for_each(dense_vec.begin(), dense_vec.end(), [this, root](const std::string dense){
				auto s = root["dense_shape"][dense].size();
				std::vector<size_t> shape;	
				for(unsigned int i = 0; i < s; ++i)
					shape.push_back(root["dense_shape"][dense][i].asInt());
				dense_shape.push_back(shape);
			});
			PARSE_VALUE(batch_size, Int)
			PARSE_VALUE(class_num, Int)
			PARSE_VALUE(model_name, String)
			PARSE_VALUE(pb_path, String)
		}
	}
private:
	std::vector<std::string> dense_vec, gradient_vec, observe_vec;
	std::vector<std::vector<size_t>> dense_shape;
	const std::string json_path;
	std::string pb_path;
	size_t batch_size;
	size_t class_num;
	std::vector<size_t> input_size_vec;
	std::vector<size_t> output_size_vec;
	std::string model_name;
};

std::ostream& operator<<(std::ostream& out, const model_conf& model) {
	out << "model_info:{";
	out << "model_name:" << model.model_name;
	out << ",pb_path:" << model.pb_path;
	out << ",batch_size:" << model.batch_size;
	out << ",class_num:" << model.class_num << ",";
	PRINT_V(model.dense_vec)
	PRINT_V(model.gradient_vec)
	PRINT_V(model.input_size_vec)
	PRINT_V(model.output_size_vec)
	PRINT_V(model.observe_vec)
	PRINT_VV(model.dense_shape)
	out << "}\n";
	return out;
}
