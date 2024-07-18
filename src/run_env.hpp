#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;

class TFRunEnv {
public:
	TFRunEnv(const std::string graph_path) : graph_path(graph_path) {
		// Initialize a tensorflow session
		TF_CHECK_OK(NewSession(SessionOptions(), &session));
        	// Read in the protobuf graph we exported
		TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_path, &graph_def));
        	// Add the graph to the session
        	TF_CHECK_OK(session->Create(graph_def));
	}
public:
	Session* 		session;
	GraphDef 		graph_def;
	const std::string 	graph_path;
};

