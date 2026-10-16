#include "mln/sdk/functional/broadcast.hpp"
#include "mln/sdk/functional/gated_delta_rule.hpp"
#include "mln/sdk/functional/matmul.hpp"
#include "mln/sdk/functional/reducesum.hpp"
#include "mln/sdk/operation/compatibility.hpp"
#include "syn/package.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <string_view>

namespace Compatibilities::Vulkan {
    namespace {
        bool targets_vulkan(const Mln::Sdk::Operation::Compatibility::Context &context) {
            return context.target && context.target->device == "vulkan";
        }
    }

    class Support final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &) const override { return false; }

            bool supports(const Context &context) const override {
                if (!targets_vulkan(context))
                    return false;
                using Kind = Mln::Sdk::Functional::Kind;
                switch (context.operation.kind) {
                case Kind::Const:
                case Kind::Add:
                case Kind::Subtract:
                case Kind::Multiply:
                case Kind::MatMul:
                case Kind::Attention:
                case Kind::Reshape:
                case Kind::Broadcast:
                case Kind::Concatenate:
                case Kind::Convert:
                case Kind::BitcastConvert:
                case Kind::BitwiseAnd:
                case Kind::ShiftRightLogical:
                case Kind::Iota:
                case Kind::Logistic:
                case Kind::Slice:
                case Kind::Transpose:
                case Kind::Exp:
                case Kind::Softplus:
                case Kind::ReduceSum:
                case Kind::ReduceMax:
                case Kind::Divide:
                case Kind::Equal:
                case Kind::Select:
                case Kind::Rsqrt:
                case Kind::Tanh:
                case Kind::Gather:
                case Kind::DynamicUpdateSlice:
                case Kind::Map:
                case Kind::CausalConv1D:
                case Kind::GatedDeltaRule:
                    return true;
                default:
                    return false;
                }
            }
    };

    class Attention final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return targets_vulkan(context) &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::Attention;
            }

            std::optional<bool> expand(const Context &) const override { return true; }
    };

    class Broadcast final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                if (!targets_vulkan(context) ||
                    context.operation.kind != Mln::Sdk::Functional::Kind::Exp ||
                    context.consumers.empty())
                    return false;

                const Mln::Sdk::Operation &consumer = *context.consumers.back();
                if (consumer.kind != Mln::Sdk::Functional::Kind::Broadcast)
                    return false;
                const auto &broadcast =
                    static_cast<const Mln::Sdk::Functional::Broadcast &>(consumer);
                const auto elements = [](const std::vector<int64_t> &shape) {
                    std::uint64_t result = 1;
                    for (const int64_t dimension : shape) {
                        if (dimension <= 0 || result > std::numeric_limits<std::uint64_t>::max() /
                                                           static_cast<std::uint64_t>(dimension))
                            return std::uint64_t{0};
                        result *= static_cast<std::uint64_t>(dimension);
                    }
                    return result;
                };
                const std::uint64_t source = elements(context.operation.signature());
                const std::uint64_t destination = elements(broadcast.signature());
                return source != 0 && destination / source >= 1024;
            }

            bool barrier(const Context &) const override { return true; }
    };

    class GatedDeltaRule final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return targets_vulkan(context) &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::GatedDeltaRule;
            }

            std::optional<bool> expand(const Context &) const override { return true; }
    };

    class Reduction final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                if (!targets_vulkan(context) ||
                    context.operation.kind != Mln::Sdk::Functional::Kind::ReduceSum)
                    return false;
                const auto &reduction =
                    static_cast<const Mln::Sdk::Functional::ReduceSum &>(context.operation);
                const auto rank = static_cast<int64_t>(reduction.x.shape.size());
                if (!std::ranges::any_of(reduction.axes, [rank](int64_t axis) {
                        return axis >= 0 && axis + 1 < rank;
                    }))
                    return false;
                const auto output_rank = reduction.signature().size();
                return std::ranges::any_of(context.consumers, [output_rank](const auto *op) {
                    if (op->kind != Mln::Sdk::Functional::Kind::Broadcast)
                        return false;
                    const auto &broadcast =
                        static_cast<const Mln::Sdk::Functional::Broadcast &>(*op);
                    return broadcast.shape.size() > output_rank;
                });
            }

            bool barrier(const Context &) const override { return true; }
    };

    class MatMul final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return targets_vulkan(context) &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::MatMul;
            }

            std::optional<bool> linearize_reduction(const Context &) const override { return true; }

            std::optional<int64_t> contraction_tile(const Context &) const override { return 64; }
    };

}

PUBLISH(vulkan_attention, Compatibilities::Vulkan::Attention{})
PUBLISH(vulkan_support, Compatibilities::Vulkan::Support{})
PUBLISH(vulkan_broadcast, Compatibilities::Vulkan::Broadcast{})
PUBLISH(vulkan_gated_delta_rule, Compatibilities::Vulkan::GatedDeltaRule{})
PUBLISH(vulkan_reduction, Compatibilities::Vulkan::Reduction{})
PUBLISH(vulkan_matmul, Compatibilities::Vulkan::MatMul{})
