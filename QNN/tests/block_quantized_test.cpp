#include "qnn.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <print>

int main() {
    using namespace Compatibilities::Qnn;
    Compilation compilation;
    if (!compilation.ready()) {
        std::println(stderr, "{}", compilation.error());
        return 1;
    }
    const std::array<std::uint32_t, 2> input_shape{1, 32};
    const std::array<std::uint32_t, 2> weight_shape{1, 32};
    const std::array<std::uint32_t, 2> output_shape{1, 1};
    const std::array<float, 1> scales{0.125f};
    const std::array<std::int32_t, 1> offsets{-8};

    const auto add_graph = [&](std::string_view name, std::span<const std::byte> codes,
                               std::uint32_t bitwidth) -> bool {
        auto graph = compilation.graph(name);
        if (!graph) {
            std::println(stderr, "{}", graph.error());
            return false;
        }
        auto input = compilation.tensor(*graph, "input", QNN_TENSOR_TYPE_APP_WRITE,
                                        QNN_DATATYPE_FLOAT_16, input_shape);
        auto weight = compilation.block_quantized_tensor(*graph, "weight", weight_shape, codes,
                                                         scales, offsets, 32, bitwidth);
        auto output = compilation.tensor(*graph, "output", QNN_TENSOR_TYPE_APP_READ,
                                         QNN_DATATYPE_FLOAT_16, output_shape);
        if (!input || !weight || !output) {
            std::println(stderr, "tensor creation failed: {} {} {}", input ? "ok" : input.error(),
                         weight ? "ok" : weight.error(), output ? "ok" : output.error());
            return false;
        }
        std::array<Qnn_Tensor_t, 2> inputs{*input, *weight};
        std::array<Qnn_Tensor_t, 1> outputs{*output};
        auto node = compilation.node(*graph, "fc", QNN_OP_FULLY_CONNECTED, inputs, outputs);
        if (!node) {
            std::println(stderr, "{}", node.error());
            return false;
        }
        auto finalized = compilation.finalize(*graph);
        if (!finalized) {
            std::println(stderr, "{}", finalized.error());
            return false;
        }
        return true;
    };

    const std::array<std::byte, 16> q4_codes{};
    const std::array<std::byte, 32> q8_codes{};
    if (!add_graph("q4_block_test", q4_codes, 4) || !add_graph("q8_block_test", q8_codes, 8))
        return 1;
    auto binary = compilation.binary();
    if (!binary) {
        std::println(stderr, "{}", binary.error());
        return 1;
    }
    std::println("QNN U4/I8 block context: {} bytes", binary->size());
}
