#include "ol_trainer.cpp"
#include "../utils/common.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;
using namespace tol;

template <typename T1, typename T2>
void wrapMyTemplateClass(py::module& m, const std::string& className) {
    py::class_<Trainer<T1, T2>>(m, className.c_str())
        .def(py::init<string, Parameter_backend, int, uint32_t, int>())
        .def("initializer", &Trainer<T1, T2>::init_weight)
	.def("train_online", &Trainer<T1, T2>::train_online)
	.def("predict", &Trainer<T1, T2>::predict);
}

void wrapTemplateClasses(py::module& m) {
    wrapMyTemplateClass<uint16_t, uint16_t>(m, "forRec");
    wrapMyTemplateClass<float, float>(m, "forImage");
}

PYBIND11_MODULE(libpytrain, m) {
    m.doc() = "pybind11";
    wrapTemplateClasses(m);
    py::enum_<Parameter_backend>(m, "Parameter_backend")
        .value("IO", Parameter_backend::IO)
        .value("PS", Parameter_backend::PS)
	.export_values();
}

// ref : https://cloud.tencent.com/developer/information/%E5%A6%82%E4%BD%95%E7%94%A8pybind11%E5%8C%85%E8%A3%85%E6%A8%A1%E6%9D%BF%E5%8C%96%E7%9A%84%E7%B1%BB-article
