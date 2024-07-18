#ifndef OL_UTILS
#define OL_UTILS

#include<iostream>
#include<algorithm>
#include<vector>
#include<fstream>
#include<sstream>
#include<chrono>
#include<iomanip>

#define OUTLOG(message) std::cout << "\E[34m" <<  __FILE__ << ":" << __FUNCTION__ << "[" << __LINE__ << "] INFO \33[0m" << message << std::endl;
#define OUTERR(message) std::cout << "\E[31;1m" <<  __FILE__ << ":" << __FUNCTION__ << "[" << __LINE__ << "] ERROR \33[0m" << message << std::endl;
#define TS(s) std::to_string(s)                                                                                                                                     
#define TS2(s) #s + string(":") + TS(s) 

#define MODEL_CONF_PATH "../model/model_conf/"
#define DOT_JSON        ".json"

enum Parameter_backend {IO = 0, PS = 1};

std::vector<std::string> stringSplit(const std::string& str, const char delim) {
	std::stringstream ss(str);
	std::string item;
	std::vector<std::string> elems;
	while (std::getline(ss, item, delim)) elems.push_back(item);
	if (str.at(str.size() - 1) == delim) elems.push_back("");
    	return elems;
}

uint16_t get_fid(uint16_t slot_id, std::string value) {
        return (slot_id << 10) | (std::hash<std::string>()(value) & 0x3ff);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
        os << "[";
        for(size_t i = 0; i < vec.size(); ++i) {
                os << vec[i] << ",]"[i == vec.size() - 1];
        }
        return os;
}

std::string current_timesteamp() {
	auto now = std::chrono::system_clock::now(); 
    	std::time_t in_time_t = std::chrono::system_clock::to_time_t(now);
	std::tm* local_tm = std::localtime(&in_time_t);
	std::stringstream ss;
	return ss.str();
}

class read_single_img {
public:
        read_single_img(const std::string &path) : path(path) {}
        size_t size() const {return data.size();}
        std::vector<float> operator()() {
                std::ifstream fin(this->path);
                if (!fin)  throw std::runtime_error("open error" + this->path);
                std::string line, token;
                while (getline(fin, line)) {
                        std::istringstream is(line);
                        while (is >> token) {
                                if (token == "[[" || token == "]]" || token == "[" || token == "]")
                                        continue;
                                int pos;
                                if((pos = token.find(',')) > 0) token.replace(pos, 1, "");
                                if((pos = token.find("]]")) > 0) token.replace(pos, 2, "");
                                data.push_back(atof(token.c_str()));
                        }
                }
                for_each(data.begin(), data.end(), [](float &d) {d/=255.0;});
                fin.close();
                return data;
        }
private:
        const std::string path;
        std::vector<float> data;
};

struct image_norm {
public:
        image_norm(float fact) : fact(fact) {}
        void operator()(float &value) {
                value /= fact;
        }
private:
        float fact;
};

#endif
