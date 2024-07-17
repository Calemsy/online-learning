#include "ws_train.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

/*
class mnist_trainer: public ws_train {
public:
        mnist_trainer(const string& model_name) 
		: ws_train("../model/model_conf/" + model_name + ".json"), model_name(model_name) {}

        void run_train_offline(size_t epoch = 100) override {
                train_offline(new mnist_dataset(), epoch);
        }
        // ~mnist_train() = default;
private:
        const string json_path = "../model/model_conf/";
	const string model_name;
};
*/

PYBIND11_MODULE(libpytrain, m) {
        m.doc() = "pybind11";
        pybind11::class_<ws_train>(m, "ws_train")
                .def(pybind11::init<string>())
                .def("initializer", &ws_train::init_weight)
                .def("train_online", &ws_train::train_online)
                .def("predict", &ws_train::predict);
                //.def("train_offline", &mnist_train::run_train_offline);
}
