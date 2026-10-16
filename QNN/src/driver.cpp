#include "mln/sdk/driver/module.hpp"

#include <QnnInterface.h>
#include <QnnMem.h>
#include <System/QnnSystemInterface.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Compatibilities::Qnn {
    using Module = Mln::Sdk::Driver::Module;
    using Function = Mln::Sdk::Driver::Function;
    using Binding = Mln::Sdk::Driver::Binding;

    namespace {
        void fail(char *error, std::size_t capacity, const char *message) {
            if (error && capacity)
                std::snprintf(error, capacity, "%s", message);
        }

        template <typename... Arguments>
        void fail(char *error, std::size_t capacity, const char *format, Arguments... arguments) {
            if (error && capacity)
                std::snprintf(error, capacity, format, arguments...);
        }

        class Library {
            public:
                explicit Library(const char *name)
                    : handle_(::dlopen(name, RTLD_NOW | RTLD_LOCAL)) {}
                ~Library() {
                    if (handle_)
                        ::dlclose(handle_);
                }
                Library(const Library &) = delete;
                Library &operator=(const Library &) = delete;

                template <typename Function> Function symbol(const char *name) const {
                    return reinterpret_cast<Function>(::dlsym(handle_, name));
                }

                [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }

            private:
                void *handle_ = nullptr;
        };

        struct RegisteredBuffer {
                void *data = nullptr;
                Qnn_MemHandle_t handle = nullptr;
                std::uint64_t size = 0;
        };

        struct Graph {
                std::string name;
                Qnn_GraphHandle_t handle = nullptr;
                std::vector<Qnn_Tensor_t> inputs;
                std::vector<Qnn_Tensor_t> outputs;
                std::uint32_t binding_count = 0;
                struct Slice {
                        std::uint32_t ordinal = 0;
                        std::uint64_t offset = 0;
                        std::uint64_t length = 0;
                };
                std::vector<Slice> input_slices;
                std::vector<Slice> output_slices;
                std::vector<RegisteredBuffer> input_buffers;
                std::vector<RegisteredBuffer> output_buffers;
        };

        struct Mapping {
                std::string name;
                std::uint32_t binding_count = 0;
                std::vector<Graph::Slice> inputs;
                std::vector<Graph::Slice> outputs;
        };

        bool read(const std::uint8_t *&data, std::size_t &size, std::uint32_t &value) {
            if (size < sizeof(value))
                return false;
            value = 0;
            for (unsigned shift = 0; shift != 32; shift += 8)
                value |= static_cast<std::uint32_t>(*data++) << shift;
            size -= sizeof(value);
            return true;
        }

        bool read(const std::uint8_t *&data, std::size_t &size, std::uint64_t &value) {
            if (size < sizeof(value))
                return false;
            value = 0;
            for (unsigned shift = 0; shift != 64; shift += 8)
                value |= static_cast<std::uint64_t>(*data++) << shift;
            size -= sizeof(value);
            return true;
        }

        bool read(const std::uint8_t *&data, std::size_t &size,
                  std::vector<std::uint32_t> &values) {
            std::uint32_t count = 0;
            if (!read(data, size, count) || count > size / sizeof(std::uint32_t))
                return false;
            values.resize(count);
            for (std::uint32_t &value : values)
                if (!read(data, size, value))
                    return false;
            return true;
        }

        bool read(const std::uint8_t *&data, std::size_t &size, std::vector<Graph::Slice> &values) {
            std::uint32_t count = 0;
            if (!read(data, size, count))
                return false;
            values.resize(count);
            for (Graph::Slice &value : values)
                if (!read(data, size, value.ordinal) || !read(data, size, value.offset) ||
                    !read(data, size, value.length))
                    return false;
            return true;
        }

        struct Executable {
                using RpcMemAlloc = void *(*)(int, std::uint32_t, int);
                using RpcMemFree = void (*)(void *);
                using RpcMemToFd = int (*)(void *);

                Library backend_library{"libQnnHtp.so"};
                Library system_library{"libQnnSystem.so"};
                Library rpc_library{"libcdsprpc.so"};
                RpcMemAlloc rpcmem_alloc = nullptr;
                RpcMemFree rpcmem_free = nullptr;
                RpcMemToFd rpcmem_to_fd = nullptr;
                const QnnInterface_t *backend_provider = nullptr;
                const QnnSystemInterface_t *system_provider = nullptr;
                Qnn_BackendHandle_t backend = nullptr;
                Qnn_DeviceHandle_t device = nullptr;
                Qnn_ContextHandle_t context = nullptr;
                QnnSystemContext_Handle_t system_context = nullptr;
                std::vector<Graph> graphs;

                ~Executable() {
                    if (backend_provider) {
                        const auto &api = backend_provider->QNN_INTERFACE_VER_NAME;
                        for (Graph &graph : graphs) {
                            const auto release = [&](std::vector<RegisteredBuffer> &buffers) {
                                for (RegisteredBuffer &buffer : buffers) {
                                    if (buffer.handle)
                                        api.memDeRegister(&buffer.handle, 1);
                                    if (buffer.data && rpcmem_free)
                                        rpcmem_free(buffer.data);
                                }
                            };
                            release(graph.input_buffers);
                            release(graph.output_buffers);
                        }
                    }
                    if (context && backend_provider)
                        backend_provider->QNN_INTERFACE_VER_NAME.contextFree(context, nullptr);
                    if (device && backend_provider)
                        backend_provider->QNN_INTERFACE_VER_NAME.deviceFree(device);
                    if (backend && backend_provider)
                        backend_provider->QNN_INTERFACE_VER_NAME.backendFree(backend);
                    if (system_context && system_provider)
                        system_provider->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(
                            system_context);
                }
        };

        struct GraphView {
                const char *name = nullptr;
                std::uint32_t input_count = 0;
                Qnn_Tensor_t *inputs = nullptr;
                std::uint32_t output_count = 0;
                Qnn_Tensor_t *outputs = nullptr;
        };

        GraphView graph_view(const QnnSystemContext_GraphInfo_t &graph) {
            switch (graph.version) {
            case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
                return {graph.graphInfoV1.graphName, graph.graphInfoV1.numGraphInputs,
                        graph.graphInfoV1.graphInputs, graph.graphInfoV1.numGraphOutputs,
                        graph.graphInfoV1.graphOutputs};
            case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
                return {graph.graphInfoV2.graphName, graph.graphInfoV2.numGraphInputs,
                        graph.graphInfoV2.graphInputs, graph.graphInfoV2.numGraphOutputs,
                        graph.graphInfoV2.graphOutputs};
            case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
                return {graph.graphInfoV3.graphName, graph.graphInfoV3.numGraphInputs,
                        graph.graphInfoV3.graphInputs, graph.graphInfoV3.numGraphOutputs,
                        graph.graphInfoV3.graphOutputs};
            default:
                return {};
            }
        }

        std::pair<std::uint32_t, QnnSystemContext_GraphInfo_t *>
        graphs(const QnnSystemContext_BinaryInfo_t &information) {
            switch (information.version) {
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
                return {information.contextBinaryInfoV1.numGraphs,
                        information.contextBinaryInfoV1.graphs};
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
                return {information.contextBinaryInfoV2.numGraphs,
                        information.contextBinaryInfoV2.graphs};
            case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
                return {information.contextBinaryInfoV3.numGraphs,
                        information.contextBinaryInfoV3.graphs};
            default:
                return {};
            }
        }

        struct TensorView {
                std::uint32_t rank = 0;
                std::uint32_t *dimensions = nullptr;
                Qnn_DataType_t data_type = QNN_DATATYPE_UNDEFINED;
        };

        TensorView tensor_view(const Qnn_Tensor_t &tensor) {
            switch (tensor.version) {
            case QNN_TENSOR_VERSION_1:
                return {tensor.v1.rank, tensor.v1.dimensions, tensor.v1.dataType};
            case QNN_TENSOR_VERSION_2:
                return {tensor.v2.rank, tensor.v2.dimensions, tensor.v2.dataType};
            default:
                return {};
            }
        }

        bool set_mem_handle(Qnn_Tensor_t &tensor, Qnn_MemHandle_t handle) {
            switch (tensor.version) {
            case QNN_TENSOR_VERSION_1:
                tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
                tensor.v1.memHandle = handle;
                return true;
            case QNN_TENSOR_VERSION_2:
                tensor.v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
                tensor.v2.memHandle = handle;
                return true;
            default:
                return false;
            }
        }

        bool register_buffer(Executable &executable, Qnn_Tensor_t &tensor, std::uint64_t size,
                             RegisteredBuffer &buffer, char *error, std::size_t error_capacity) {
            if (!size || size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                fail(error, error_capacity, "QNN tensor buffer size is invalid");
                return false;
            }
            const TensorView view = tensor_view(tensor);
            if (view.data_type == QNN_DATATYPE_UNDEFINED || (view.rank != 0 && !view.dimensions)) {
                fail(error, error_capacity, "QNN tensor metadata version is unsupported");
                return false;
            }

            constexpr int system_heap = 25;
            constexpr std::uint32_t default_flags = 1;
            buffer.data =
                executable.rpcmem_alloc(system_heap, default_flags, static_cast<int>(size));
            if (!buffer.data) {
                fail(error, error_capacity, "QNN rpcmem allocation failed");
                return false;
            }
            buffer.size = size;
            const int descriptor = executable.rpcmem_to_fd(buffer.data);
            if (descriptor < 0) {
                fail(error, error_capacity, "QNN rpcmem file descriptor is invalid");
                return false;
            }

            Qnn_MemDescriptor_t memory = QNN_MEM_DESCRIPTOR_INIT;
            memory.memShape.numDim = view.rank;
            memory.memShape.dimSize = view.dimensions;
            memory.dataType = view.data_type;
            memory.memType = QNN_MEM_TYPE_ION;
            memory.ionInfo.fd = descriptor;
            const auto &api = executable.backend_provider->QNN_INTERFACE_VER_NAME;
            const Qnn_ErrorHandle_t status =
                api.memRegister(executable.context, &memory, 1, &buffer.handle);
            if (status != QNN_SUCCESS || !buffer.handle) {
                fail(error, error_capacity, "QNN rpcmem registration failed (%u)",
                     static_cast<unsigned>(status));
                return false;
            }
            if (!set_mem_handle(tensor, buffer.handle)) {
                fail(error, error_capacity, "QNN tensor metadata version is unsupported");
                return false;
            }
            return true;
        }

        void *load(const std::uint8_t *data, std::size_t size, char *error,
                   std::size_t error_capacity) {
            constexpr char magic[] = "MLNQNN2\0";
            if (size < 8 || std::memcmp(data, magic, 8) != 0) {
                fail(error, error_capacity, "QNN executable header is invalid");
                return nullptr;
            }
            data += 8;
            size -= 8;
            std::uint64_t binary_size = 0;
            std::uint32_t mapping_count = 0;
            if (!read(data, size, binary_size) || !read(data, size, mapping_count)) {
                fail(error, error_capacity, "QNN executable header is truncated");
                return nullptr;
            }
            std::vector<Mapping> mappings;
            mappings.reserve(mapping_count);
            for (std::uint32_t index = 0; index < mapping_count; ++index) {
                std::uint32_t name_size = 0;
                Mapping mapping;
                if (!read(data, size, name_size) || name_size > size) {
                    fail(error, error_capacity, "QNN executable mapping name is invalid");
                    return nullptr;
                }
                mapping.name.assign(reinterpret_cast<const char *>(data), name_size);
                data += name_size;
                size -= name_size;
                if (!read(data, size, mapping.binding_count) || !read(data, size, mapping.inputs) ||
                    !read(data, size, mapping.outputs)) {
                    fail(error, error_capacity, "QNN executable mapping is truncated");
                    return nullptr;
                }
                mappings.push_back(std::move(mapping));
            }
            if (binary_size != size) {
                fail(error, error_capacity, "QNN executable payload size is invalid");
                return nullptr;
            }

            auto executable = std::make_unique<Executable>();
            if (!executable->backend_library || !executable->system_library ||
                !executable->rpc_library) {
                fail(error, error_capacity, "failed to load QNN libraries: %s", ::dlerror());
                return nullptr;
            }
            executable->rpcmem_alloc =
                executable->rpc_library.symbol<Executable::RpcMemAlloc>("rpcmem_alloc");
            executable->rpcmem_free =
                executable->rpc_library.symbol<Executable::RpcMemFree>("rpcmem_free");
            executable->rpcmem_to_fd =
                executable->rpc_library.symbol<Executable::RpcMemToFd>("rpcmem_to_fd");
            if (!executable->rpcmem_alloc || !executable->rpcmem_free ||
                !executable->rpcmem_to_fd) {
                fail(error, error_capacity, "FastRPC shared-buffer symbols are missing");
                return nullptr;
            }

            using BackendProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
            using SystemProviders =
                Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t ***, uint32_t *);
            const auto backend_providers =
                executable->backend_library.symbol<BackendProviders>("QnnInterface_getProviders");
            const auto system_providers = executable->system_library.symbol<SystemProviders>(
                "QnnSystemInterface_getProviders");
            if (!backend_providers || !system_providers) {
                fail(error, error_capacity, "QNN provider symbols are missing");
                return nullptr;
            }

            const QnnInterface_t **backend_list = nullptr;
            const QnnSystemInterface_t **system_list = nullptr;
            std::uint32_t backend_count = 0;
            std::uint32_t system_count = 0;
            if (backend_providers(&backend_list, &backend_count) != QNN_SUCCESS ||
                backend_count == 0 ||
                system_providers(&system_list, &system_count) != QNN_SUCCESS || system_count == 0) {
                fail(error, error_capacity, "QNN did not provide a compatible interface");
                return nullptr;
            }
            executable->backend_provider = backend_list[0];
            executable->system_provider = system_list[0];
            const auto &api = executable->backend_provider->QNN_INTERFACE_VER_NAME;
            const auto &system = executable->system_provider->QNN_SYSTEM_INTERFACE_VER_NAME;

            if (api.backendCreate(nullptr, nullptr, &executable->backend) != QNN_SUCCESS ||
                api.deviceCreate(nullptr, nullptr, &executable->device) != QNN_SUCCESS ||
                system.systemContextCreate(&executable->system_context) != QNN_SUCCESS) {
                fail(error, error_capacity, "QNN initialization failed");
                return nullptr;
            }

            const QnnSystemContext_BinaryInfo_t *binary_information = nullptr;
            if (system.systemContextGetMetaData(executable->system_context, data, size,
                                                &binary_information) != QNN_SUCCESS ||
                !binary_information) {
                fail(error, error_capacity, "QNN context metadata is invalid");
                return nullptr;
            }
            if (api.contextCreateFromBinary(executable->backend, executable->device, nullptr, data,
                                            size, &executable->context, nullptr) != QNN_SUCCESS) {
                fail(error, error_capacity, "QNN context restoration failed");
                return nullptr;
            }

            const auto [graph_count, graph_information] = graphs(*binary_information);
            if (!graph_count || !graph_information) {
                fail(error, error_capacity, "QNN context contains no graphs");
                return nullptr;
            }
            executable->graphs.reserve(graph_count);
            for (std::uint32_t index = 0; index < graph_count; ++index) {
                const GraphView view = graph_view(graph_information[index]);
                if (!view.name) {
                    fail(error, error_capacity, "QNN graph metadata version is unsupported");
                    return nullptr;
                }
                Graph graph;
                graph.name = view.name;
                graph.inputs.assign(view.inputs, view.inputs + view.input_count);
                graph.outputs.assign(view.outputs, view.outputs + view.output_count);
                if (api.graphRetrieve(executable->context, graph.name.c_str(), &graph.handle) !=
                    QNN_SUCCESS) {
                    fail(error, error_capacity, "QNN graph '%s' could not be retrieved",
                         graph.name.c_str());
                    return nullptr;
                }
                const auto mapping = std::ranges::find_if(
                    mappings, [&](const Mapping &value) { return value.name == graph.name; });
                if (mapping == mappings.end() || mapping->inputs.size() != graph.inputs.size() ||
                    mapping->outputs.size() != graph.outputs.size()) {
                    fail(error, error_capacity, "QNN graph '%s' binding map is invalid",
                         graph.name.c_str());
                    return nullptr;
                }
                graph.binding_count = mapping->binding_count;
                graph.input_slices = mapping->inputs;
                graph.output_slices = mapping->outputs;
                graph.input_buffers.resize(graph.inputs.size());
                graph.output_buffers.resize(graph.outputs.size());
                executable->graphs.push_back(std::move(graph));
                Graph &stored = executable->graphs.back();
                for (std::size_t tensor = 0; tensor < stored.inputs.size(); ++tensor)
                    if (!register_buffer(*executable, stored.inputs[tensor],
                                         stored.input_slices[tensor].length,
                                         stored.input_buffers[tensor], error, error_capacity))
                        return nullptr;
                for (std::size_t tensor = 0; tensor < stored.outputs.size(); ++tensor)
                    if (!register_buffer(*executable, stored.outputs[tensor],
                                         stored.output_slices[tensor].length,
                                         stored.output_buffers[tensor], error, error_capacity))
                        return nullptr;
            }
            return executable.release();
        }

        void destroy(void *value) { delete static_cast<Executable *>(value); }

        std::size_t function_count(void *value) {
            return static_cast<Executable *>(value)->graphs.size();
        }

        std::int32_t function(void *value, std::size_t ordinal, Function *result, char *error,
                              std::size_t error_capacity) {
            auto &graphs = static_cast<Executable *>(value)->graphs;
            if (ordinal >= graphs.size()) {
                fail(error, error_capacity, "QNN graph ordinal is out of range");
                return 1;
            }
            Graph &graph = graphs[ordinal];
            *result = {graph.name.c_str(), 0, static_cast<std::uint16_t>(graph.binding_count)};
            return 0;
        }

        std::int32_t invoke(void *value, std::size_t ordinal, const std::uint32_t *, std::size_t,
                            const Binding *bindings, std::size_t binding_count, char *error,
                            std::size_t error_capacity) {
            auto &executable = *static_cast<Executable *>(value);
            if (ordinal >= executable.graphs.size()) {
                fail(error, error_capacity, "QNN graph ordinal is out of range");
                return 1;
            }
            Graph &graph = executable.graphs[ordinal];
            if (binding_count != graph.binding_count) {
                fail(error, error_capacity, "QNN graph expected %zu bindings but received %zu",
                     graph.inputs.size() + graph.outputs.size(), binding_count);
                return 1;
            }
            for (std::size_t index = 0; index < graph.inputs.size(); ++index) {
                const Graph::Slice &slice = graph.input_slices[index];
                const Binding &binding = bindings[slice.ordinal];
                if (!binding.data || slice.offset > binding.size ||
                    slice.length > binding.size - slice.offset) {
                    fail(error, error_capacity, "QNN input slice is outside its binding");
                    return 1;
                }
                std::memcpy(graph.input_buffers[index].data,
                            static_cast<std::uint8_t *>(binding.data) + slice.offset, slice.length);
            }
            for (std::size_t index = 0; index < graph.outputs.size(); ++index) {
                const Graph::Slice &slice = graph.output_slices[index];
                const Binding &binding = bindings[slice.ordinal];
                if (!binding.data || slice.offset > binding.size ||
                    slice.length > binding.size - slice.offset) {
                    fail(error, error_capacity, "QNN output slice is outside its binding");
                    return 1;
                }
            }
            const auto &api = executable.backend_provider->QNN_INTERFACE_VER_NAME;
            if (api.graphExecute(graph.handle, graph.inputs.data(), graph.inputs.size(),
                                 graph.outputs.data(), graph.outputs.size(), nullptr,
                                 nullptr) != QNN_SUCCESS) {
                fail(error, error_capacity, "QNN graph execution failed");
                return 1;
            }
            for (std::size_t index = 0; index < graph.outputs.size(); ++index) {
                const Graph::Slice &slice = graph.output_slices[index];
                const Binding &binding = bindings[slice.ordinal];
                std::memcpy(static_cast<std::uint8_t *>(binding.data) + slice.offset,
                            graph.output_buffers[index].data, slice.length);
            }
            return 0;
        }
    }

    const Module Driver{
        .abi_version = Mln::Sdk::Driver::AbiVersion,
        .struct_size = sizeof(Module),
        .name = "qnn",
        .format = "qnn-context-v2",
        .load = load,
        .destroy = destroy,
        .function_count = function_count,
        .function = function,
        .invoke = invoke,
        .option_count = 0,
        .options = nullptr,
        .configure = nullptr,
    };
}

extern "C" const Mln::Sdk::Driver::Module *mln_runtime_driver() {
    return &Compatibilities::Qnn::Driver;
}
