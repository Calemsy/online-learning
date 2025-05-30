// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: train_serving.proto
#ifndef GRPC_train_5fserving_2eproto__INCLUDED
#define GRPC_train_5fserving_2eproto__INCLUDED

#include "train_serving.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/method_handler_impl.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace grpc_impl {
class CompletionQueue;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc_impl

namespace grpc {
namespace experimental {
template <typename RequestT, typename ResponseT>
class MessageAllocator;
}  // namespace experimental
}  // namespace grpc

namespace wt {

// 指定服务的名称，作为生成代码里面的二级namespace
class Caca final {
 public:
  static constexpr char const* service_full_name() {
    return "wt.Caca";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status score(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::wt::PredictReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::PredictReply>> Asyncscore(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::PredictReply>>(AsyncscoreRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::PredictReply>> PrepareAsyncscore(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::PredictReply>>(PrepareAsyncscoreRaw(context, request, cq));
    }
    virtual ::grpc::Status training(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::wt::TrainReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::TrainReply>> Asynctraining(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::TrainReply>>(AsynctrainingRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::TrainReply>> PrepareAsynctraining(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::wt::TrainReply>>(PrepareAsynctrainingRaw(context, request, cq));
    }
    class experimental_async_interface {
     public:
      virtual ~experimental_async_interface() {}
      virtual void score(::grpc::ClientContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void score(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::PredictReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void score(::grpc::ClientContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      virtual void score(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::PredictReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      virtual void training(::grpc::ClientContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void training(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::TrainReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void training(::grpc::ClientContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      virtual void training(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::TrainReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
    };
    virtual class experimental_async_interface* experimental_async() { return nullptr; }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::wt::PredictReply>* AsyncscoreRaw(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::wt::PredictReply>* PrepareAsyncscoreRaw(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::wt::TrainReply>* AsynctrainingRaw(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::wt::TrainReply>* PrepareAsynctrainingRaw(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status score(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::wt::PredictReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::PredictReply>> Asyncscore(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::PredictReply>>(AsyncscoreRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::PredictReply>> PrepareAsyncscore(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::PredictReply>>(PrepareAsyncscoreRaw(context, request, cq));
    }
    ::grpc::Status training(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::wt::TrainReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::TrainReply>> Asynctraining(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::TrainReply>>(AsynctrainingRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::TrainReply>> PrepareAsynctraining(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::wt::TrainReply>>(PrepareAsynctrainingRaw(context, request, cq));
    }
    class experimental_async final :
      public StubInterface::experimental_async_interface {
     public:
      void score(::grpc::ClientContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response, std::function<void(::grpc::Status)>) override;
      void score(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::PredictReply* response, std::function<void(::grpc::Status)>) override;
      void score(::grpc::ClientContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      void score(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::PredictReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      void training(::grpc::ClientContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response, std::function<void(::grpc::Status)>) override;
      void training(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::TrainReply* response, std::function<void(::grpc::Status)>) override;
      void training(::grpc::ClientContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      void training(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::wt::TrainReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
     private:
      friend class Stub;
      explicit experimental_async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class experimental_async_interface* experimental_async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class experimental_async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::wt::PredictReply>* AsyncscoreRaw(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::wt::PredictReply>* PrepareAsyncscoreRaw(::grpc::ClientContext* context, const ::wt::PredictRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::wt::TrainReply>* AsynctrainingRaw(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::wt::TrainReply>* PrepareAsynctrainingRaw(::grpc::ClientContext* context, const ::wt::TrainRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_score_;
    const ::grpc::internal::RpcMethod rpcmethod_training_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response);
    virtual ::grpc::Status training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_score : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_score() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_score() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requestscore(::grpc::ServerContext* context, ::wt::PredictRequest* request, ::grpc::ServerAsyncResponseWriter< ::wt::PredictReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_training : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_training() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_training() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requesttraining(::grpc::ServerContext* context, ::wt::TrainRequest* request, ::grpc::ServerAsyncResponseWriter< ::wt::TrainReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_score<WithAsyncMethod_training<Service > > AsyncService;
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_score : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithCallbackMethod_score() {
      ::grpc::Service::experimental().MarkMethodCallback(0,
        new ::grpc_impl::internal::CallbackUnaryHandler< ::wt::PredictRequest, ::wt::PredictReply>(
          [this](::grpc::ServerContext* context,
                 const ::wt::PredictRequest* request,
                 ::wt::PredictReply* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   return this->score(context, request, response, controller);
                 }));
    }
    void SetMessageAllocatorFor_score(
        ::grpc::experimental::MessageAllocator< ::wt::PredictRequest, ::wt::PredictReply>* allocator) {
      static_cast<::grpc_impl::internal::CallbackUnaryHandler< ::wt::PredictRequest, ::wt::PredictReply>*>(
          ::grpc::Service::experimental().GetHandler(0))
              ->SetMessageAllocator(allocator);
    }
    ~ExperimentalWithCallbackMethod_score() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_training : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithCallbackMethod_training() {
      ::grpc::Service::experimental().MarkMethodCallback(1,
        new ::grpc_impl::internal::CallbackUnaryHandler< ::wt::TrainRequest, ::wt::TrainReply>(
          [this](::grpc::ServerContext* context,
                 const ::wt::TrainRequest* request,
                 ::wt::TrainReply* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   return this->training(context, request, response, controller);
                 }));
    }
    void SetMessageAllocatorFor_training(
        ::grpc::experimental::MessageAllocator< ::wt::TrainRequest, ::wt::TrainReply>* allocator) {
      static_cast<::grpc_impl::internal::CallbackUnaryHandler< ::wt::TrainRequest, ::wt::TrainReply>*>(
          ::grpc::Service::experimental().GetHandler(1))
              ->SetMessageAllocator(allocator);
    }
    ~ExperimentalWithCallbackMethod_training() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  typedef ExperimentalWithCallbackMethod_score<ExperimentalWithCallbackMethod_training<Service > > ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_score : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_score() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_score() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_training : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_training() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_training() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_score : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithRawMethod_score() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_score() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requestscore(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawMethod_training : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithRawMethod_training() {
      ::grpc::Service::MarkMethodRaw(1);
    }
    ~WithRawMethod_training() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requesttraining(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_score : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithRawCallbackMethod_score() {
      ::grpc::Service::experimental().MarkMethodRawCallback(0,
        new ::grpc_impl::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
          [this](::grpc::ServerContext* context,
                 const ::grpc::ByteBuffer* request,
                 ::grpc::ByteBuffer* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   this->score(context, request, response, controller);
                 }));
    }
    ~ExperimentalWithRawCallbackMethod_score() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void score(::grpc::ServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_training : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithRawCallbackMethod_training() {
      ::grpc::Service::experimental().MarkMethodRawCallback(1,
        new ::grpc_impl::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
          [this](::grpc::ServerContext* context,
                 const ::grpc::ByteBuffer* request,
                 ::grpc::ByteBuffer* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   this->training(context, request, response, controller);
                 }));
    }
    ~ExperimentalWithRawCallbackMethod_training() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void training(::grpc::ServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_score : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_score() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< ::wt::PredictRequest, ::wt::PredictReply>(std::bind(&WithStreamedUnaryMethod_score<BaseClass>::Streamedscore, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_score() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status score(::grpc::ServerContext* context, const ::wt::PredictRequest* request, ::wt::PredictReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status Streamedscore(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::wt::PredictRequest,::wt::PredictReply>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_training : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_training() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler< ::wt::TrainRequest, ::wt::TrainReply>(std::bind(&WithStreamedUnaryMethod_training<BaseClass>::Streamedtraining, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_training() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status training(::grpc::ServerContext* context, const ::wt::TrainRequest* request, ::wt::TrainReply* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status Streamedtraining(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::wt::TrainRequest,::wt::TrainReply>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_score<WithStreamedUnaryMethod_training<Service > > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_score<WithStreamedUnaryMethod_training<Service > > StreamedService;
};

}  // namespace wt


#endif  // GRPC_train_5fserving_2eproto__INCLUDED
