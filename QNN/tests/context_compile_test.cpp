#include "qnn.hpp"

#include <QnnOpDef.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2)
        throw std::invalid_argument("context_compile_test OUTPUT");

    Compatibilities::Qnn::Compilation compilation;
    if (!compilation.ready())
        throw std::runtime_error("QNN compilation API is unavailable");
    auto graph = compilation.graph("qnn_add");
    if (!graph)
        throw std::runtime_error(graph.error());

    const std::uint32_t shape[]{1, 4};
    auto left = compilation.quantized_tensor(*graph, "left", QNN_TENSOR_TYPE_APP_WRITE,
                                             QNN_DATATYPE_UFIXED_POINT_8, shape, 1.0F);
    auto right = compilation.quantized_tensor(*graph, "right", QNN_TENSOR_TYPE_APP_WRITE,
                                              QNN_DATATYPE_UFIXED_POINT_8, shape, 1.0F);
    auto output = compilation.quantized_tensor(*graph, "output", QNN_TENSOR_TYPE_APP_READ,
                                               QNN_DATATYPE_UFIXED_POINT_8, shape, 1.0F);
    if (!left || !right || !output)
        throw std::runtime_error("QNN rejected test tensors");

    Qnn_Tensor_t inputs[]{*left, *right};
    Qnn_Tensor_t outputs[]{*output};
    auto node = compilation.node(*graph, "add", QNN_OP_ELEMENT_WISE_ADD, inputs, outputs);
    if (!node)
        throw std::runtime_error(node.error());
    auto finalized = compilation.finalize(*graph);
    if (!finalized)
        throw std::runtime_error(finalized.error());
    auto binary = compilation.binary();
    if (!binary)
        throw std::runtime_error(binary.error());

    std::ofstream stream(argv[1], std::ios::binary);
    stream.write(reinterpret_cast<const char *>(binary->data()),
                 static_cast<std::streamsize>(binary->size()));
    if (!stream)
        throw std::runtime_error("failed to write QNN context");
}
