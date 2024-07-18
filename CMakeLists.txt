cmake_minimum_required(VERSION 2.8.8)

project(onlearning)

add_definitions(-w)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 20)

aux_source_directory(./proto DIR_SRCS) 

# find_package(PkgConfig REQUIRED)
# pkg_search_module(protobuf REQUIRED protobuf)
find_package(Python 3.6.5 REQUIRED COMPONENTS Interpreter Development)
find_package(Protobuf REQUIRED)
if (Protobuf_FOUND)
    message(STATUS "Found Protobuf version: ${Protobuf_VERSION}")
else()
    message(FATAL_ERROR "Could not find Protobuf")
endif()

# -I
include_directories(
		/usr/local/include
		./third_part/includes
		./third_part/includes/output_base/external/eigen_archive
		./third_part/includes/abseil-cpp
)

# -L
link_directories(/usr/local/lib ./third_part/libs)

# -o
add_executable(tol_server ../src/server.cpp ${DIR_SRCS})
add_executable(tol_client ../src/client.cpp ${DIR_SRCS})
add_library(pytrain SHARED ../src/bindpy.cpp)
add_executable(ps_server ../ps/ps_server.cpp)

install(TARGETS pytrain LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/bin)
install(TARGETS tol_server DESTINATION ${PROJECT_SOURCE_DIR}/bin)
install(TARGETS tol_client DESTINATION ${PROJECT_SOURCE_DIR}/bin)
install(TARGETS ps_server DESTINATION ${PROJECT_SOURCE_DIR}/bin)

# -l
target_link_libraries(tol_server 
				libtensorflow_cc.so 
				libtensorflow_framework.so 
				grpc++_reflection 
				grpc++ 
				dl 
				pthread 
				grpc_unsecure 
				address_sorting 
				gpr 
				cares 
				z 
				ps
				protobuf
				zmq
				boost_program_options)
target_link_libraries(tol_client
			 	libtensorflow_cc.so 
				libtensorflow_framework.so 
				grpc++_reflection 
				grpc++ 
				dl 
				pthread 
				grpc_unsecure 
				address_sorting 
				gpr 
				cares 
				z 
				boost_program_options)
target_link_libraries(pytrain 	libtensorflow_cc.so 
				libtensorflow_framework.so
				ps
				zmq
				Python::Python
				jsoncpp)
target_link_libraries(ps_server ps
				boost_program_options
				protobuf
				zmq
				pthread)
