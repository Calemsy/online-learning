#ifndef OL_TENSOR
#define OL_TENSOR
#include "tensorflow/core/framework/tensor.h"
#include <vector>

using namespace tensorflow;
using namespace std;

template <typename T>
Tensor make_tensor(const vector<T>& data, const vector<size_t>& dims) {
        TensorShape t_shape;
        size_t i = 0;
        for_each(dims.begin(), dims.end(), [&i, &t_shape](int dim){t_shape.InsertDim(i++, dim);});
        Tensor tensor(DataTypeToEnum<T>::value, t_shape);
        copy_n(data.data(), data.size(), tensor.flat<T>().data());
        return tensor;
}

template <typename T1, typename T2>
class MakeIOTensor {
public:
        MakeIOTensor(const vector<T1>& x_data, const vector<size_t>&& x_shape, const vector<T2>& y_data, const vector<size_t>& y_shape)
                        : x_data(x_data)
                        , x_shape(x_shape)
                        , y_data(y_data)
                        , y_shape(y_shape) {
                x = make_tensor<T1>(this->x_data, this->x_shape);
                y = make_tensor<T2>(this->y_data, this->y_shape);
        }
        Tensor x, y;
private:
        const vector<size_t> x_shape;
        const vector<size_t> y_shape;
        const vector<T1> x_data;
        const vector<T2> y_data;
};
#endif
