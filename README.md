## onlearning

### 0. What Is This?

> This is a project at the "toy" level.

This project has implemented a computational service framework that supports real-time training and prediction based on user-defined models in the scenario of streaming data.

The most suitable scenario for this project is when you want to conduct real-time training of models using streaming data.

All the user needs to do is to customize the TensorFlow network diagram based on the Python template provided in the project according to their own requirements. The rest of the work is what this project is dedicated to achieving automatically.

### 1. Project Architecture

`TensorFlow` defines the network structure and exports the `pb` graph along with a configuration file that describe the network parameters (already implemented in the python template).

`C++` uses the `tf api` to load the network graph, construct the runtime environment, and generate a `trainer`.

This `trainer` obtains samples (`x` and `y`) from the real-time stream, pulls the already trained (or initialization) weight parameters from the parameter server(ps-lite or simple relay on file) respectively, performs forward calculation for `loss`, backward calculation for gradients, and pushes the gradients to the parameter server to update the parameters.

The above functions such as reading the model, populating data, calculating gradients, and interacting with the `ps` are encapsulated and implemented by `c++`, and an invokable interface is provided through `grpc`. Therefore, you can consume data from `kafka` through a `flink` task, and after accumulating enough data for a `batch` in the counting window, call the `grpc` service provided by `c++` to perform real-time online learning of the model.

![](https://www.helloimg.com/i/2024/11/24/6741fd26aacab.png)

### 2. How to Use

#### 2.1、project compilation

```bash
mkdir build && cd build
cmake..
make
make install
```

#### 2.2、demo

The following is a simple demonstration case. Using `ps-lite` as the backend for parameter storage (starting 1 server-node and 2 worker nodes), the MLP model is trained.

```bash
# each instruction is executed in a new terminal.
# 1、start the parameter server service
./ps_server -r scheduler -s 1 -w 2
./ps_server -r server -s 1 -w 2
# 2、start the server side, specify to use the above-mentioned parameter server to manage parameter synchronization, and set the number of workers to 2 (task parallelism).
./tol_server -p ps -n 2
# 3、start the first worker. 
./tol_client -m mlp -e 10
# 4、start the second worker.
./tol_client -m mlp -e 10
# ...
# 5、use linux signal to send instructions so that the parameter will saved into a file when the service is exited.
ps aux | grep ps_server | awk '{print $2}' | xargs kill -15
# 6、exit completely
ps aux | grep ps_server | awk '{print $2}' | xargs kill -9
```



