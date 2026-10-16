#include "mln/sdk/backend/compatibility.hpp"
#include "qnn.hpp"
#include "syn/package.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <mutex>
#include <print>
#include <span>
#include <string>
#include <vector>

namespace Compatibilities::Qnn {
    namespace {
        using Binding = Mln::Sdk::Backend::Compilation::Binding;
        using Element = Mln::Sdk::Backend::Compilation::Element;
        using Entrypoint = Mln::Sdk::Backend::Compilation::Entrypoint;

        float half_to_float(std::uint16_t value) {
            const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000) << 16;
            std::uint32_t exponent = (value >> 10) & 0x1f;
            std::uint32_t mantissa = value & 0x3ff;
            if (exponent == 0) {
                if (mantissa == 0)
                    return std::bit_cast<float>(sign);
                unsigned shift = 0;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    ++shift;
                }
                mantissa &= 0x3ff;
                exponent = 127 - 15 + 1 - shift;
            } else if (exponent == 0x1f) {
                exponent = 0xff;
            } else {
                exponent += 127 - 15;
            }
            return std::bit_cast<float>(sign | (exponent << 23) | (mantissa << 13));
        }

        struct Mapping {
                struct Slice {
                        std::uint32_t ordinal = 0;
                        std::uint64_t offset = 0;
                        std::uint64_t length = 0;
                };
                std::string name;
                std::uint32_t binding_count = 0;
                std::vector<Slice> inputs;
                std::vector<Slice> outputs;
        };

        const Mln::Sdk::Backend::Resource *
        resource(const Mln::Sdk::Backend::Compilation &compilation, std::string_view entrypoint,
                 std::uint32_t ordinal) {
            const auto relation =
                std::ranges::find_if(compilation.bindings, [&](const auto &binding) {
                    return binding.entrypoint == entrypoint && binding.ordinal == ordinal;
                });
            if (relation == compilation.bindings.end())
                return nullptr;
            const auto value = std::ranges::find_if(compilation.resources, [&](const auto &item) {
                return item.name == relation->resource;
            });
            return value == compilation.resources.end() ? nullptr : &*value;
        }

        const Mln::Sdk::Backend::Resource *
        resource(const Mln::Sdk::Backend::Compilation &compilation, const Binding &binding,
                 std::string_view suffix) {
            if (const auto *value = resource(compilation, binding.entrypoint, binding.ordinal))
                return value;
            std::size_t elements = 1;
            for (const std::int64_t dimension : binding.shape) {
                if (dimension <= 0)
                    return nullptr;
                elements *= static_cast<std::size_t>(dimension);
            }
            const auto found = std::ranges::find_if(compilation.resources, [&](const auto &value) {
                if (!value.name.ends_with(suffix))
                    return false;
                if (suffix == ".scale")
                    return value.data.size() == elements * sizeof(float) ||
                           value.data.size() == elements * sizeof(std::uint16_t);
                return value.data.size() == elements;
            });
            return found == compilation.resources.end() ? nullptr : &*found;
        }

        std::expected<Qnn_DataType_t, std::string> dtype(Element type) {
            switch (type) {
            case Element::I4:
                return QNN_DATATYPE_SFIXED_POINT_4;
            case Element::U4:
                return QNN_DATATYPE_UFIXED_POINT_4;
            case Element::I8:
                return QNN_DATATYPE_INT_8;
            case Element::I16:
                return QNN_DATATYPE_INT_16;
            case Element::I32:
                return QNN_DATATYPE_INT_32;
            case Element::F16:
                return QNN_DATATYPE_FLOAT_16;
            case Element::F32:
                return QNN_DATATYPE_FLOAT_32;
            case Element::BF16:
                return std::unexpected("QNN does not support BF16 dispatch bindings");
            }
            return std::unexpected("QNN does not support the dispatch element type");
        }

        std::expected<std::vector<std::uint32_t>, std::string>
        shape(std::span<const std::int64_t> dimensions) {
            std::vector<std::uint32_t> result;
            result.reserve(dimensions.size());
            for (const std::int64_t dimension : dimensions) {
                if (dimension <= 0 || dimension > UINT32_MAX)
                    return std::unexpected("QNN dispatch binding dimension is invalid");
                result.push_back(static_cast<std::uint32_t>(dimension));
            }
            return result;
        }

        std::expected<Mapping, std::string> compile(Compilation &compilation,
                                                    const Entrypoint &entrypoint,
                                                    const Mln::Sdk::Backend::Compilation &context) {
            if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace) {
                std::println(stderr, "qnn graph {}: {} resources", entrypoint.name,
                             context.resources.size());
                for (const auto &value : context.resources) {
                    if (value.name.contains("layers.0.linear_attn.in_proj_qkv"))
                        std::println(stderr, "  resource {} format={} rank={} bytes={}", value.name,
                                     value.format, value.shape.size(), value.data.size());
                }
                for (const auto &value : context.bindings) {
                    if (value.entrypoint == entrypoint.name)
                        std::println(
                            stderr,
                            "  binding {} element={} rank={} read={} written={} resource={}",
                            value.ordinal, static_cast<int>(value.element), value.shape.size(),
                            value.read, value.written, value.resource);
                }
            }
            if (entrypoint.contractions > 1)
                return std::unexpected(
                    std::format("QNN graph '{}' supports at most one contraction, found {}",
                                entrypoint.name, entrypoint.contractions));

            std::vector<const Binding *> descriptions;
            for (const Binding &binding : context.bindings) {
                if (binding.entrypoint != entrypoint.name)
                    continue;
                const auto existing = std::ranges::find_if(descriptions, [&](const Binding *value) {
                    return value->ordinal == binding.ordinal && value->offset == binding.offset;
                });
                if (existing == descriptions.end()) {
                    descriptions.push_back(&binding);
                } else if ((*existing)->resource.empty() && !binding.resource.empty()) {
                    *existing = &binding;
                }
            }
            if (descriptions.empty())
                return std::unexpected("QNN dispatch has no tensor bindings");
            const auto quantized = std::ranges::find_if(descriptions, [&](const Binding *binding) {
                if (binding->resource.empty())
                    return false;
                const auto *value = resource(context, *binding, ".q");
                if (!value || !value->name.ends_with(".q"))
                    return false;

                return binding->element == Element::I8 || binding->element == Element::I4 ||
                       binding->element == Element::U4;
            });

            const bool fully_connected =
                entrypoint.contractions == 1 && quantized != descriptions.end();
            const Binding *scale_binding = nullptr;
            const Binding *activation_binding = nullptr;
            if (fully_connected) {
                const auto *weight = resource(context, **quantized, ".q");
                const std::string base =
                    std::string(weight->name.substr(0, weight->name.size() - 2));
                const auto scale = std::ranges::find_if(descriptions, [&](const Binding *binding) {
                    const auto *value = resource(context, *binding, ".scale");
                    return value && value->name == base + ".scale";
                });
                if (scale != descriptions.end())
                    scale_binding = *scale;
                if (!scale_binding && !weight->shape.empty()) {
                    const auto channels = weight->shape.front();
                    const auto adjacent =
                        std::ranges::find_if(descriptions, [&](const Binding *binding) {
                            return binding->ordinal == (*quantized)->ordinal + 1 && binding->read &&
                                   !binding->written && binding->shape.size() == 1 &&
                                   binding->shape.front() == channels &&
                                   (binding->element == Element::F16 ||
                                    binding->element == Element::F32);
                        });
                    if (adjacent != descriptions.end())
                        scale_binding = *adjacent;
                }
                if (weight->shape.size() >= 2) {
                    std::int64_t input_channels = 1;
                    for (std::size_t index = 1; index < weight->shape.size(); ++index) {
                        const std::int64_t dimension = weight->shape[index];
                        if (dimension <= 0 ||
                            input_channels > std::numeric_limits<std::int64_t>::max() / dimension) {
                            input_channels = 0;
                            break;
                        }
                        input_channels *= dimension;
                    }
                    const auto transposed_input_channels = weight->shape.front();
                    const auto activation =
                        std::ranges::find_if(descriptions, [&](const Binding *binding) {
                            return binding != *quantized && binding != scale_binding &&
                                   binding->read && !binding->written && !binding->shape.empty() &&
                                   (binding->shape.back() == input_channels ||
                                    binding->shape.back() == transposed_input_channels) &&
                                   (binding->element == Element::F16 ||
                                    binding->element == Element::F32);
                        });
                    if (activation != descriptions.end())
                        activation_binding = *activation;
                }
            }
            const std::size_t reads =
                std::ranges::count_if(descriptions, [&](const Binding *binding) {
                    return binding->read && !binding->written &&
                           (!fully_connected || binding == activation_binding);
                });
            const std::size_t writes = std::ranges::count_if(
                descriptions, [](const Binding *binding) { return binding->written; });
            if (fully_connected && activation_binding) {
                const auto output = std::ranges::find_if(
                    descriptions, [](const Binding *binding) { return binding->written; });
                if (output != descriptions.end() &&
                    (*output)->element != activation_binding->element)
                    return std::unexpected(std::format(
                        "QNN mixed-precision FullyConnected '{}' requires its output to match "
                        "the activation dtype",
                        entrypoint.name));
            }
            const std::size_t expected_reads = fully_connected                ? 1
                                               : entrypoint.contractions == 1 ? 2
                                                                              : 1;
            if (reads != expected_reads || writes != 1) {
                const std::size_t q_resources =
                    std::ranges::count_if(context.resources, [](const auto &value) {
                        return value.name.ends_with(".q");
                    });
                const std::size_t scale_resources =
                    std::ranges::count_if(context.resources, [](const auto &value) {
                        return value.name.ends_with(".scale");
                    });
                std::string candidates;
                for (const auto &value : context.resources) {
                    if (!value.name.contains("layers.0.linear_attn.in_proj_qkv"))
                        continue;
                    candidates += std::format(" [{} format={} bytes={} shape={}]", value.name,
                                              value.format, value.data.size(), value.shape.size());
                }
                return std::unexpected(std::format(
                    "QNN graph '{}' requires {} input binding(s) and one output binding "
                    "(reads={}, writes={}, quantized={}, scale={}, q_resources={}, "
                    "scale_resources={}){}",
                    entrypoint.name, expected_reads, reads, writes, fully_connected,
                    scale_binding != nullptr, q_resources, scale_resources, candidates));
            }

            auto graph = compilation.graph(entrypoint.name);
            if (!graph)
                return std::unexpected(graph.error());
            Mapping mapping{.name = entrypoint.name};
            for (const Binding *binding : descriptions)
                mapping.binding_count = std::max(mapping.binding_count, binding->ordinal + 1);
            std::vector<Qnn_Tensor_t> inputs;
            std::vector<Qnn_Tensor_t> outputs;
            for (const Binding *binding : descriptions) {
                if (fully_connected && !binding->written && binding != activation_binding)
                    continue;
                auto dimensions = shape(binding->shape);
                auto element = dtype(binding->element);
                if (!dimensions)
                    return std::unexpected(dimensions.error());
                if (!element)
                    return std::unexpected(element.error());
                const bool output = binding->written;
                const auto tensor_name =
                    std::format("binding_{}_{}", binding->ordinal, binding->offset);
                auto tensor = binding->element == Element::I8
                                  ? compilation.quantized_tensor(*graph, tensor_name,
                                                                 output ? QNN_TENSOR_TYPE_APP_READ
                                                                        : QNN_TENSOR_TYPE_APP_WRITE,
                                                                 QNN_DATATYPE_SFIXED_POINT_8,
                                                                 *dimensions, 1.0F)
                                  : compilation.tensor(*graph, tensor_name,
                                                       output ? QNN_TENSOR_TYPE_APP_READ
                                                              : QNN_TENSOR_TYPE_APP_WRITE,
                                                       *element, *dimensions);
                if (!tensor)
                    return std::unexpected(tensor.error());
                if (output) {
                    outputs.push_back(*tensor);
                    mapping.outputs.push_back({binding->ordinal, binding->offset, binding->length});
                } else {
                    inputs.push_back(*tensor);
                    mapping.inputs.push_back({binding->ordinal, binding->offset, binding->length});
                }
            }
            if (fully_connected) {
                const auto *weight = resource(context, **quantized, ".q");
                const std::string base =
                    std::string(weight->name.substr(0, weight->name.size() - 2));
                const auto scale = std::ranges::find_if(context.resources, [&](const auto &value) {
                    return value.name == base + ".scale";
                });
                if (scale == context.resources.end() ||
                    (scale->data.size() % sizeof(float) != 0 &&
                     scale->data.size() % sizeof(std::uint16_t) != 0))
                    return std::unexpected("QNN affine weight is missing its scale tensor");
                std::vector<float> scales;
                if (scale->data.size() ==
                    static_cast<std::size_t>(weight->shape.front()) * sizeof(float)) {
                    scales.resize(scale->data.size() / sizeof(float));
                    std::memcpy(scales.data(), scale->data.data(), scale->data.size());
                } else {
                    scales.resize(scale->data.size() / sizeof(std::uint16_t));
                    for (std::size_t index = 0; index < scales.size(); ++index) {
                        std::uint16_t value = 0;
                        std::memcpy(&value, scale->data.data() + index * sizeof(value),
                                    sizeof(value));
                        scales[index] = half_to_float(value);
                    }
                }
                std::vector<std::uint32_t> storage_shape;
                storage_shape.reserve(weight->shape.size());
                for (const std::int64_t dimension : weight->shape) {
                    if (dimension <= 0 || dimension > UINT32_MAX)
                        return std::unexpected("QNN quantized weight has an invalid shape");
                    storage_shape.push_back(static_cast<std::uint32_t>(dimension));
                }
                std::expected<Qnn_Tensor_t, std::string> tensor =
                    std::unexpected("unsupported QNN quantized weight semantics");
                if ((*quantized)->element == Element::I8 || (*quantized)->element == Element::I4 ||
                    (*quantized)->element == Element::U4) {
                    const auto zero =
                        std::ranges::find_if(context.resources, [&](const auto &value) {
                            return value.name == base + ".zero";
                        });
                    if (zero == context.resources.end() || zero->data.size() != scale->data.size())
                        return std::unexpected(
                            "QNN block-affine weight is missing its zero-point tensor");
                    std::vector<std::int32_t> offsets(scales.size());
                    for (std::size_t index = 0; index < offsets.size(); ++index) {
                        std::uint16_t bits = 0;
                        std::memcpy(&bits, zero->data.data() + index * sizeof(bits), sizeof(bits));
                        const float zero_point = half_to_float(bits);
                        const auto integral = std::lround(zero_point);
                        if (std::abs(zero_point - static_cast<float>(integral)) > 0.0001F)
                            return std::unexpected(
                                "QNN block-affine weight has a non-integral zero point");
                        offsets[index] = -static_cast<std::int32_t>(integral);
                    }
                    if (weight->shape.size() < 3 || weight->shape.back() <= 0)
                        return std::unexpected("QNN block-affine weight has no block dimension");
                    std::uint64_t input_channels = 1;
                    for (std::size_t index = 1; index < storage_shape.size(); ++index) {
                        if (input_channels > UINT32_MAX / storage_shape[index])
                            return std::unexpected(
                                "QNN block-affine U4 logical shape overflows uint32");
                        input_channels *= storage_shape[index];
                    }
                    const std::array<std::uint32_t, 2> logical_shape{
                        storage_shape.front(), static_cast<std::uint32_t>(input_channels)};
                    tensor = compilation.block_quantized_tensor(
                        *graph, "weight", logical_shape, weight->data, scales, offsets,
                        static_cast<std::uint32_t>(weight->shape.back()),
                        (*quantized)->element == Element::I8 ? 8U : 4U);
                }
                if (!tensor)
                    return std::unexpected(tensor.error());
                inputs.push_back(*tensor);
            }
            const char *type = fully_connected                ? QNN_OP_FULLY_CONNECTED
                               : entrypoint.contractions == 1 ? QNN_OP_MAT_MUL
                                                              : QNN_OP_CAST;
            auto node = compilation.node(*graph, fully_connected ? "fully_connected" : "operation",
                                         type, inputs, outputs);
            if (!node)
                return std::unexpected(node.error());
            auto finalized = compilation.finalize(*graph);
            if (!finalized)
                return std::unexpected(finalized.error());
            return mapping;
        }

        void append(std::vector<std::byte> &output, std::uint32_t value) {
            for (unsigned shift = 0; shift != 32; shift += 8)
                output.push_back(static_cast<std::byte>((value >> shift) & 0xff));
        }

        void append(std::vector<std::byte> &output, std::uint64_t value) {
            for (unsigned shift = 0; shift != 64; shift += 8)
                output.push_back(static_cast<std::byte>((value >> shift) & 0xff));
        }
    }

    class Backend final : public Mln::Sdk::Backend::Compatibility {
        public:
            std::string_view name() const noexcept override { return "qnn"; }
            std::string_view format() const noexcept override { return "qnn-context-v2"; }

            Legality legality(const Context &context) const override {
                const bool feeds_contraction =
                    std::ranges::any_of(context.consumers, [](const auto *consumer) {
                        return consumer && consumer->kind == Mln::Sdk::Functional::Kind::MatMul;
                    });
                const bool weight_preparation =
                    context.operation.kind == Mln::Sdk::Functional::Kind::Map ||
                    context.operation.kind == Mln::Sdk::Functional::Kind::Reshape ||
                    context.operation.kind == Mln::Sdk::Functional::Kind::Transpose ||
                    context.operation.kind == Mln::Sdk::Functional::Kind::Convert ||
                    context.operation.kind == Mln::Sdk::Functional::Kind::Broadcast ||
                    context.operation.kind == Mln::Sdk::Functional::Kind::Multiply;
                return context.operation.kind == Mln::Sdk::Functional::Kind::MatMul ||
                               context.operation.kind == Mln::Sdk::Functional::Kind::Const ||
                               (weight_preparation && feeds_contraction)
                           ? Legality::Exact
                           : Legality::Unsupported;
            }

            bool barrier(const Context &context) const override {
                return context.operation.kind == Mln::Sdk::Functional::Kind::MatMul;
            }

            std::optional<bool> fuse_producer(const Context &context) const override {
                if (context.operation.kind == Mln::Sdk::Functional::Kind::MatMul)
                    return false;
                return std::nullopt;
            }

            int priority() const override { return 1000; }

            std::expected<std::vector<std::byte>, std::string>
            serialize(const Mln::Sdk::Backend::Compilation &context) const override {
                static std::mutex compiler_mutex;
                const std::lock_guard lock(compiler_mutex);
                Compilation compilation;
                if (!compilation.ready())
                    return std::unexpected(std::format(
                        "QNN host compilation API is unavailable: {}", compilation.error()));

                std::vector<Mapping> mappings;
                for (const Entrypoint &entrypoint : context.entrypoints) {
                    auto result = compile(compilation, entrypoint, context);
                    if (!result)
                        return std::unexpected(result.error());
                    mappings.push_back(std::move(*result));
                }
                auto binary = compilation.binary();
                if (!binary)
                    return binary;
                std::vector<std::byte> result;
                constexpr std::string_view magic{"MLNQNN2\0", 8};
                result.insert(result.end(), reinterpret_cast<const std::byte *>(magic.data()),
                              reinterpret_cast<const std::byte *>(magic.data() + magic.size()));
                append(result, static_cast<std::uint64_t>(binary->size()));
                append(result, static_cast<std::uint32_t>(mappings.size()));
                for (const Mapping &mapping : mappings) {
                    append(result, static_cast<std::uint32_t>(mapping.name.size()));
                    result.insert(result.end(),
                                  reinterpret_cast<const std::byte *>(mapping.name.data()),
                                  reinterpret_cast<const std::byte *>(mapping.name.data() +
                                                                      mapping.name.size()));
                    append(result, mapping.binding_count);
                    append(result, static_cast<std::uint32_t>(mapping.inputs.size()));
                    for (const Mapping::Slice &slice : mapping.inputs) {
                        append(result, slice.ordinal);
                        append(result, slice.offset);
                        append(result, slice.length);
                    }
                    append(result, static_cast<std::uint32_t>(mapping.outputs.size()));
                    for (const Mapping::Slice &slice : mapping.outputs) {
                        append(result, slice.ordinal);
                        append(result, slice.offset);
                        append(result, slice.length);
                    }
                }
                result.insert(result.end(), binary->begin(), binary->end());
                return result;
            }
    };
}

PUBLISH(qnn_backend, Compatibilities::Qnn::Backend{})
