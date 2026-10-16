#pragma once

#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <QnnInterface.h>
#include <QnnOpDef.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <dlfcn.h>
#include <expected>
#include <link.h>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Compatibilities::Qnn {
    class Library {
        public:
            explicit Library(std::string name, Lmid_t name_space = LM_ID_BASE,
                             int flags = RTLD_NOW | RTLD_LOCAL)
                : value_(::dlmopen(name_space, name.c_str(), flags)) {}
            ~Library() {
                if (value_)
                    ::dlclose(value_);
            }

            Library(const Library &) = delete;
            Library &operator=(const Library &) = delete;

            template <typename Function> [[nodiscard]] Function symbol(const char *name) const {
                return reinterpret_cast<Function>(::dlsym(value_, name));
            }

            [[nodiscard]] explicit operator bool() const noexcept { return value_ != nullptr; }

            [[nodiscard]] Lmid_t name_space() const noexcept {
                Lmid_t result = LM_ID_BASE;
                return value_ && ::dlinfo(value_, RTLD_DI_LMID, &result) == 0 ? result : LM_ID_BASE;
            }

        private:
            void *value_ = nullptr;
    };

    inline std::string library_path(const char *environment, std::string_view suffix,
                                    std::string_view fallback) {
        const char *root = std::getenv(environment);
        if (!root || !*root)
            return std::string(fallback);
        std::string result(root);
        if (!result.ends_with('/'))
            result.push_back('/');
        result.append(suffix);
        return result;
    }

    class Compilation {
        public:
            Compilation()
                : unwind_(library_path("QAIRT_HOST_RUNTIME", "libunwind.so.1", "libunwind.so.1"),
                          LM_ID_NEWLM),
                  abi_(library_path("QAIRT_HOST_RUNTIME", "libc++abi.so.1", "libc++abi.so.1"),
                       unwind_.name_space()),
                  cpp_(library_path("QAIRT_HOST_RUNTIME", "libc++.so.1", "libc++.so.1"),
                       unwind_.name_space()),
                  library_(library_path("QAIRT_ROOT", "lib/x86_64-linux-clang/libQnnHtp.so",
                                        "libQnnHtp.so"),
                           unwind_.name_space()) {
                if (!library_) {
                    const char *message = ::dlerror();
                    error_ = message ? message : "failed to load libQnnHtp.so";
                    return;
                }
                using Providers = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
                const auto providers = library_.symbol<Providers>("QnnInterface_getProviders");
                const QnnInterface_t **values = nullptr;
                std::uint32_t count = 0;
                if (!providers || providers(&values, &count) != QNN_SUCCESS || count == 0) {
                    error_ = "QNN did not provide a host backend interface";
                    return;
                }
                provider_ = values[0];
                const auto &api = provider_->QNN_INTERFACE_VER_NAME;
                if (api.backendCreate(nullptr, nullptr, &backend_) != QNN_SUCCESS) {
                    error_ = "QNN failed to create the host backend";
                    return;
                }
                const QnnDevice_Config_t **device_configs = nullptr;
                QnnHtpDevice_CustomConfig_t soc_config{};
                QnnDevice_Config_t device_config = QNN_DEVICE_CONFIG_INIT;
                const QnnDevice_Config_t *offline_device_configs[] = {&device_config, nullptr};
                if (std::getenv("QAIRT_ROOT")) {
                    if (const char *model = std::getenv("QAIRT_SOC_MODEL"); model && *model) {
                        soc_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
                        soc_config.socModel =
                            static_cast<std::uint32_t>(std::strtoul(model, nullptr, 0));
                    } else {
                        soc_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
                        soc_config.arch.deviceId = 0;
                        soc_config.arch.arch = QNN_HTP_DEVICE_ARCH_V68;
                    }
                    device_config.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
                    device_config.customConfig = &soc_config;
                    device_configs = offline_device_configs;
                }
                if (api.deviceCreate(nullptr, device_configs, &device_) != QNN_SUCCESS) {
                    error_ = "QNN failed to create the HTP device";
                    return;
                }
                if (api.contextCreate(backend_, device_, nullptr, &context_) != QNN_SUCCESS) {
                    error_ = "QNN failed to create the compilation context";
                    return;
                }
                ready_ = true;
            }

            ~Compilation() {
                if (!provider_)
                    return;
                const auto &api = provider_->QNN_INTERFACE_VER_NAME;
                if (context_)
                    api.contextFree(context_, nullptr);
                if (device_)
                    api.deviceFree(device_);
                if (backend_)
                    api.backendFree(backend_);
            }

            Compilation(const Compilation &) = delete;
            Compilation &operator=(const Compilation &) = delete;

            [[nodiscard]] bool ready() const noexcept { return ready_; }
            [[nodiscard]] std::string_view error() const noexcept { return error_; }

            [[nodiscard]] std::expected<Qnn_GraphHandle_t, std::string>
            graph(std::string_view name) const {
                Qnn_GraphHandle_t value = nullptr;
                const std::string owned(name);
                QnnHtpGraph_CustomConfig_t htp_config{};
                htp_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB;

                htp_config.vtcmSizeInMB = 0;
                QnnGraph_Config_t graph_config = QNN_GRAPH_CONFIG_INIT;
                graph_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
                graph_config.customConfig = &htp_config;
                const QnnGraph_Config_t *graph_configs[] = {&graph_config, nullptr};
                const QnnGraph_Config_t **configs =
                    std::getenv("QAIRT_ROOT") ? graph_configs : nullptr;
                if (api().graphCreate(context_, owned.c_str(), configs, &value) != QNN_SUCCESS)
                    return std::unexpected("QNN failed to create graph '" + owned + "'");
                return value;
            }

            [[nodiscard]] std::expected<Qnn_Tensor_t, std::string>
            tensor(Qnn_GraphHandle_t graph, std::string_view name, Qnn_TensorType_t type,
                   Qnn_DataType_t dtype, std::span<const std::uint32_t> shape,
                   std::span<const std::byte> data = {}) {
                names_.emplace_back(name);
                shapes_.emplace_back(shape.begin(), shape.end());
                Qnn_Tensor_t value = QNN_TENSOR_INIT;
                value.version = QNN_TENSOR_VERSION_2;
                value.v2 = QNN_TENSOR_V2_INIT;
                value.v2.name = names_.back().c_str();
                value.v2.type = type;
                value.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                value.v2.dataType = dtype;
                value.v2.rank = static_cast<std::uint32_t>(shapes_.back().size());
                value.v2.dimensions = shapes_.back().data();
                if (!data.empty()) {
                    value.v2.memType = QNN_TENSORMEMTYPE_RAW;
                    value.v2.clientBuf.data = const_cast<std::byte *>(data.data());
                    value.v2.clientBuf.dataSize = static_cast<std::uint32_t>(data.size());
                }
                if (api().tensorCreateGraphTensor(graph, &value) != QNN_SUCCESS)
                    return std::unexpected("QNN rejected tensor '" + std::string(name) + "'");
                return value;
            }

            [[nodiscard]] std::expected<Qnn_Tensor_t, std::string>
            quantized_tensor(Qnn_GraphHandle_t graph, std::string_view name, Qnn_TensorType_t type,
                             Qnn_DataType_t dtype, std::span<const std::uint32_t> shape,
                             float scale, std::int32_t offset = 0) {
                names_.emplace_back(name);
                shapes_.emplace_back(shape.begin(), shape.end());
                Qnn_Tensor_t value = QNN_TENSOR_INIT;
                value.version = QNN_TENSOR_VERSION_2;
                value.v2 = QNN_TENSOR_V2_INIT;
                value.v2.name = names_.back().c_str();
                value.v2.type = type;
                value.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                value.v2.dataType = dtype;
                value.v2.rank = static_cast<std::uint32_t>(shapes_.back().size());
                value.v2.dimensions = shapes_.back().data();
                value.v2.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
                value.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
                value.v2.quantizeParams.quantizationEncoding =
                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                value.v2.quantizeParams.scaleOffsetEncoding.scale = scale;
                value.v2.quantizeParams.scaleOffsetEncoding.offset = offset;
                if (api().tensorCreateGraphTensor(graph, &value) != QNN_SUCCESS)
                    return std::unexpected("QNN rejected tensor '" + std::string(name) + "'");
                return value;
            }

            [[nodiscard]] std::expected<Qnn_Tensor_t, std::string>
            axis_quantized_tensor(Qnn_GraphHandle_t graph, std::string_view name,
                                  std::span<const std::uint32_t> shape,
                                  std::span<const std::byte> data, std::span<const float> scales,
                                  std::uint32_t axis) {
                names_.emplace_back(name);
                shapes_.emplace_back(shape.begin(), shape.end());
                scale_offsets_.emplace_back();
                auto &parameters = scale_offsets_.back();
                parameters.reserve(scales.size());
                for (const float scale : scales)
                    parameters.push_back({scale, 0});
                Qnn_Tensor_t value = QNN_TENSOR_INIT;
                value.version = QNN_TENSOR_VERSION_2;
                value.v2 = QNN_TENSOR_V2_INIT;
                value.v2.name = names_.back().c_str();
                value.v2.type = QNN_TENSOR_TYPE_STATIC;
                value.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                value.v2.dataType = QNN_DATATYPE_SFIXED_POINT_8;
                value.v2.rank = static_cast<std::uint32_t>(shapes_.back().size());
                value.v2.dimensions = shapes_.back().data();
                value.v2.memType = QNN_TENSORMEMTYPE_RAW;
                value.v2.clientBuf.data = const_cast<std::byte *>(data.data());
                value.v2.clientBuf.dataSize = static_cast<std::uint32_t>(data.size());
                value.v2.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
                value.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
                value.v2.quantizeParams.quantizationEncoding =
                    QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
                value.v2.quantizeParams.axisScaleOffsetEncoding.axis = axis;
                value.v2.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets =
                    static_cast<std::uint32_t>(parameters.size());
                value.v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset = parameters.data();
                if (api().tensorCreateGraphTensor(graph, &value) != QNN_SUCCESS)
                    return std::unexpected("QNN rejected quantized tensor '" + std::string(name) +
                                           "'");
                return value;
            }

            [[nodiscard]] std::expected<Qnn_Tensor_t, std::string>
            block_quantized_tensor(Qnn_GraphHandle_t graph, std::string_view name,
                                   std::span<const std::uint32_t> shape,
                                   std::span<const std::byte> data, std::span<const float> scales,
                                   std::span<const std::int32_t> offsets,
                                   std::uint32_t group_elements, std::uint32_t bitwidth) {
                if (shape.size() != 2 || group_elements == 0 || (bitwidth != 4 && bitwidth != 8) ||
                    shape[1] % group_elements != 0 || scales.size() != offsets.size() ||
                    scales.size() !=
                        static_cast<std::size_t>(shape[0]) * (shape[1] / group_elements))
                    return std::unexpected("invalid QNN block tensor metadata");
                names_.emplace_back(name);
                shapes_.emplace_back(shape.begin(), shape.end());
                block_shapes_.push_back({1, group_elements});
                raw_buffers_.emplace_back(static_cast<std::size_t>(shape[0]) * shape[1]);
                auto &unpacked = raw_buffers_.back();
                const std::size_t expected_bytes =
                    (unpacked.size() * static_cast<std::size_t>(bitwidth) + 7) / 8;
                if (data.size() != expected_bytes)
                    return std::unexpected("invalid packed QNN block tensor length");
                if (bitwidth == 8) {
                    std::copy(data.begin(), data.end(), unpacked.begin());
                } else {
                    for (std::size_t index = 0; index < unpacked.size(); ++index) {
                        const auto packed = std::to_integer<std::uint8_t>(data[index / 2]);
                        unpacked[index] =
                            static_cast<std::byte>((packed >> ((index & 1) * 4)) & 0x0f);
                    }
                }
                scale_offsets_.emplace_back();
                auto &parameters = scale_offsets_.back();
                parameters.reserve(scales.size());
                for (std::size_t index = 0; index < scales.size(); ++index)
                    parameters.push_back({scales[index], offsets[index]});

                Qnn_Tensor_t value = QNN_TENSOR_INIT;
                value.version = QNN_TENSOR_VERSION_2;
                value.v2 = QNN_TENSOR_V2_INIT;
                value.v2.name = names_.back().c_str();
                value.v2.type = QNN_TENSOR_TYPE_STATIC;
                value.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
                value.v2.dataType = QNN_DATATYPE_UFIXED_POINT_8;
                value.v2.rank = static_cast<std::uint32_t>(shapes_.back().size());
                value.v2.dimensions = shapes_.back().data();
                value.v2.memType = QNN_TENSORMEMTYPE_RAW;
                value.v2.clientBuf.data = unpacked.data();
                value.v2.clientBuf.dataSize = static_cast<std::uint32_t>(unpacked.size());
                value.v2.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
                value.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
                value.v2.quantizeParams.quantizationEncoding =
                    QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED;
                mapped_blocks_.push_back(QNN_BW_BLOCK_MAPPED_ENCODING_INIT);
                auto &encoding = mapped_blocks_.back();
                encoding.bitwidth = bitwidth;
                encoding.mapping = QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC;
                encoding.blockSize = block_shapes_.back().data();
                encoding.scaleOffset = parameters.data();
                value.v2.quantizeParams.bwBlockMappedEncoding = &encoding;
                if (api().tensorCreateGraphTensor(graph, &value) != QNN_SUCCESS)
                    return std::unexpected("QNN rejected block tensor '" + std::string(name) + "'");
                return value;
            }

            [[nodiscard]] std::expected<void, std::string>
            node(Qnn_GraphHandle_t graph, std::string_view name, std::string_view type,
                 std::span<Qnn_Tensor_t> inputs, std::span<Qnn_Tensor_t> outputs,
                 std::span<Qnn_Param_t> parameters = {}) {
                names_.emplace_back(name);
                names_.emplace_back(type);
                Qnn_OpConfig_t operation = QNN_OPCONFIG_INIT;
                operation.version = QNN_OPCONFIG_VERSION_1;
                operation.v1 = QNN_OPCONFIG_V1_INIT;
                operation.v1.name = names_[names_.size() - 2].c_str();
                operation.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
                operation.v1.typeName = names_.back().c_str();
                operation.v1.numOfParams = static_cast<std::uint32_t>(parameters.size());
                operation.v1.params = parameters.data();
                operation.v1.numOfInputs = static_cast<std::uint32_t>(inputs.size());
                operation.v1.inputTensors = inputs.data();
                operation.v1.numOfOutputs = static_cast<std::uint32_t>(outputs.size());
                operation.v1.outputTensors = outputs.data();
                if (api().graphAddNode(graph, operation) != QNN_SUCCESS)
                    return std::unexpected("QNN rejected node '" + std::string(name) + "'");
                return {};
            }

            [[nodiscard]] std::expected<void, std::string> finalize(Qnn_GraphHandle_t graph) const {
                if (api().graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS)
                    return std::unexpected("QNN failed to finalize graph");
                return {};
            }

            [[nodiscard]] std::expected<std::vector<std::byte>, std::string> binary() const {
                Qnn_ContextBinarySize_t size = 0;
                if (api().contextGetBinarySize(context_, &size) != QNN_SUCCESS)
                    return std::unexpected("QNN failed to determine context binary size");
                std::vector<std::byte> result(static_cast<std::size_t>(size));
                Qnn_ContextBinarySize_t written = 0;
                if (api().contextGetBinary(context_, result.data(), size, &written) != QNN_SUCCESS)
                    return std::unexpected("QNN failed to serialize context binary");
                result.resize(static_cast<std::size_t>(written));
                return result;
            }

        private:
            [[nodiscard]] const QNN_INTERFACE_VER_TYPE &api() const {
                return provider_->QNN_INTERFACE_VER_NAME;
            }

            Library unwind_;
            Library abi_;
            Library cpp_;
            Library library_;
            const QnnInterface_t *provider_ = nullptr;
            Qnn_BackendHandle_t backend_ = nullptr;
            Qnn_DeviceHandle_t device_ = nullptr;
            Qnn_ContextHandle_t context_ = nullptr;
            bool ready_ = false;
            std::string error_;
            std::deque<std::string> names_;
            std::deque<std::vector<std::uint32_t>> shapes_;
            std::deque<std::vector<std::uint32_t>> block_shapes_;
            std::deque<std::vector<std::byte>> raw_buffers_;
            std::deque<std::vector<Qnn_ScaleOffset_t>> scale_offsets_;
            std::deque<Qnn_BwBlockMapped_t> mapped_blocks_;
    };
}
