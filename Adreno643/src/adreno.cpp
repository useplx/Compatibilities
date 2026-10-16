#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/functional/convert.hpp"
#include "mln/sdk/functional/matmul.hpp"
#include "mln/sdk/functional/reducesum.hpp"
#include "mln/sdk/functional/transpose.hpp"
#include "mln/sdk/operation/compatibility.hpp"
#include "syn/package.hpp"

#include <algorithm>
#include <cstdint>
#include <ranges>

namespace Compatibilities::Adreno643 {
    namespace {
        bool targets_adreno_643(const Mln::Sdk::Operation::Compatibility::Context &context) {
            if (!context.target || context.target->device != "vulkan")
                return false;
            const auto target = context.target->property("iree-vulkan-target");
            return target && *target == "adreno-643";
        }
    }

    class MatMul final : public Mln::Sdk::Operation::Compatibility {
        private:
            static constexpr int64_t WideProjectionRows = 64 * 1024;

            static bool wide(const Mln::Sdk::Functional::MatMul &matmul) {
                const std::vector<int64_t> output = matmul.signature();
                return !output.empty() && output.back() > WideProjectionRows;
            }

            static bool quantized_projection(const Mln::Sdk::Functional::MatMul &matmul) {
                if (!matmul.rhs.def ||
                    matmul.rhs.def->kind != Mln::Sdk::Functional::Kind::Transpose)
                    return false;
                const auto &transpose =
                    static_cast<const Mln::Sdk::Functional::Transpose &>(*matmul.rhs.def);
                if (transpose.permutation != std::vector<int64_t>{1, 0} || !transpose.x.def ||
                    transpose.x.def->kind != Mln::Sdk::Functional::Kind::Const)
                    return false;
                const auto &constant =
                    static_cast<const Mln::Sdk::Functional::Const &>(*transpose.x.def);
                return constant.quantize;
            }

        public:
            bool applies(const Context &context) const override {
                if (!targets_adreno_643(context) ||
                    context.operation.kind != Mln::Sdk::Functional::Kind::MatMul)
                    return false;
                const auto &matmul =
                    static_cast<const Mln::Sdk::Functional::MatMul &>(context.operation);
                const std::vector<int64_t> output = matmul.signature();
                return quantized_projection(matmul) &&
                       (matmul.lhs.shape.size() == 1 || matmul.lhs.shape.size() == 2) &&
                       std::ranges::all_of(matmul.lhs.shape,
                                           [](int64_t dimension) { return dimension > 0; }) &&
                       !output.empty() && output.back() > 0;
            }

            int64_t tile(const Context &context, int64_t axis, int64_t extent) const override {
                constexpr int64_t OutputAxis = -1;
                constexpr int64_t DecodeRows = 512;
                constexpr int64_t WideDecodeRows = 2048;
                constexpr int64_t PrefillRows = 256;
                const auto &matmul =
                    static_cast<const Mln::Sdk::Functional::MatMul &>(context.operation);
                const bool prefill = matmul.lhs.shape.size() == 2 && matmul.lhs.shape[0] > 1;
                if (!prefill && !wide(matmul))
                    return extent;
                const int64_t maximum =
                    prefill ? PrefillRows : (wide(matmul) ? WideDecodeRows : DecodeRows);
                return axis == OutputAxis ? std::min(extent, maximum) : extent;
            }

            bool unroll(const Context &) const override { return true; }

            std::optional<bool> fuse_producer(const Context &context) const override {
                const auto &matmul =
                    static_cast<const Mln::Sdk::Functional::MatMul &>(context.operation);
                return wide(matmul) || matmul.lhs.shape.size() != 1;
            }

            std::optional<bool> linearize_reduction(const Context &) const override {
                return false;
            }

            std::optional<int64_t> contraction_tile(const Context &context) const override {
                const auto &matmul =
                    static_cast<const Mln::Sdk::Functional::MatMul &>(context.operation);
                constexpr int64_t WideContraction = 512;
                return wide(matmul) ? WideContraction : matmul.lhs.shape.back();
            }

            int priority() const override { return 100; }
    };

    class Transpose final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                if (!targets_adreno_643(context) ||
                    context.operation.kind != Mln::Sdk::Functional::Kind::Transpose)
                    return false;
                const auto shape = context.operation.signature();
                if (shape.size() != 3 || shape.back() < 256)
                    return false;
                std::uint64_t elements = 1;
                for (const int64_t dimension : shape) {
                    if (dimension <= 0)
                        return false;
                    elements *= static_cast<std::uint64_t>(dimension);
                }
                return elements >= 8192;
            }

            bool barrier(const Context &) const override { return true; }
    };

    class Convert final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                if (!targets_adreno_643(context) ||
                    context.operation.kind != Mln::Sdk::Functional::Kind::Convert)
                    return false;
                const auto &convert =
                    static_cast<const Mln::Sdk::Functional::Convert &>(context.operation);
                if (convert.target != Mln::DType::Float32)
                    return false;

                std::uint64_t elements = 1;
                for (const int64_t dimension : convert.signature()) {
                    if (dimension <= 0)
                        return false;
                    elements *= static_cast<std::uint64_t>(dimension);
                }
                constexpr std::uint64_t SharedMemoryBytes = 32 * 1024;
                return elements * sizeof(float) >= SharedMemoryBytes;
            }

            bool barrier(const Context &) const override { return true; }
    };

    class Reduction final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return targets_adreno_643(context) &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::ReduceSum;
            }

            int64_t tile(const Context &, int64_t, int64_t extent) const override {

                constexpr int64_t MaximumElements = 64;
                return extent < 0 ? MaximumElements : std::min(extent, MaximumElements);
            }
    };
}

PUBLISH(adreno_matmul, Compatibilities::Adreno643::MatMul{})
PUBLISH(adreno_transpose, Compatibilities::Adreno643::Transpose{})
PUBLISH(adreno_convert, Compatibilities::Adreno643::Convert{})
PUBLISH(adreno_reduction, Compatibilities::Adreno643::Reduction{})
