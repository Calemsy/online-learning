#指定 cmake 的最小版本
cmake_minimum_required(VERSION 2.8.8)

#项目名称/工程名
project(wt)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)
set(TENSORFLOW_DIR /data0/users/shuaishuai3/wt)

# 将源码目录保存到变量中
aux_source_directory(./proto DIR_SRCS) 

find_package(PkgConfig REQUIRED)
pkg_search_module(protobuf REQUIRED protobuf)

#设置包含的目录，项目中的include路径
include_directories(
	/usr/local/include
	${TENSORFLOW_DIR}/tensorflow-1.15.0
	${TENSORFLOW_DIR}/output_base/external/eigen_archive
	${TENSORFLOW_DIR}/abseil-cpp
	./third_part/includes
)

#添加非标准动态链接库目录
link_directories(/usr/local/lib
		/data1/shiduo/anaconda2/envs/py36t/lib
		${TENSORFLOW_DIR}/grpc/cmake/build
		${TENSORFLOW_DIR}/grpc/cmake/build/third_party/cares/cares/lib
		${TENSORFLOW_DIR}/grpc/cmake/build/third_party/zlib
		${TENSORFLOW_DIR}/tensorflow-1.15.0/bazel-bin/tensorflow
		./third_part/libs
)

#生成可执行文件
add_executable(server ../src/server.cpp ${DIR_SRCS})
add_executable(client ../src/client.cpp ${DIR_SRCS})
add_library(pytrain SHARED ../src/bindpy.cpp)
install(TARGETS pytrain LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/src)
message(STATUS ${PROJECT_SOURCE_DIR}/src/)

#设置 target 需要链接的库
target_link_libraries(server libtensorflow_cc.so libtensorflow_framework.so grpc++_reflection grpc grpc++ dl pthread grpc_unsecure address_sorting gpr cares z jsoncpp python3.6m)
target_link_libraries(client libtensorflow_cc.so libtensorflow_framework.so grpc++_reflection grpc grpc++ dl pthread grpc_unsecure address_sorting gpr cares z boost_program_options)
target_link_libraries(pytrain libtensorflow_cc.so libtensorflow_framework.so grpc++_reflection grpc grpc++ dl pthread grpc_unsecure address_sorting gpr cares z jsoncpp)
