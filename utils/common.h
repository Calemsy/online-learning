#include<iostream>

std::vector<std::string> stringSplit(const std::string& str, char delim) {
	std::stringstream ss(str);
	std::string item;
	std::vector<std::string> elems;
	while (std::getline(ss, item, delim)) {
        	if (!item.empty()) {
            		elems.push_back(item);
        	}
    	}
    	return elems;
}
