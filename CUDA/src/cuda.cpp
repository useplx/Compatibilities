#include "mln/sdk/kind.hpp"
#include "mln/sdk/operation/compatibility.hpp"
#include "syn/package.hpp"

namespace Compatibilities::Cuda {
    class GatedDeltaRule final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return context.target && context.target->device == "cuda" &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::GatedDeltaRule;
            }

            std::optional<bool> expand(const Context &) const override { return true; }
    };
}

PUBLISH(cuda_gated_delta_rule, Compatibilities::Cuda::GatedDeltaRule{})
