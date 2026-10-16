#include "mln/sdk/kind.hpp"
#include "mln/sdk/operation/compatibility.hpp"
#include "syn/package.hpp"

namespace Compatibilities::Llvm {
    class Support final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &) const override { return false; }

            bool supports(const Context &context) const override {
                if (!context.target || context.target->device != "local")
                    return false;
                using Kind = Mln::Sdk::Functional::Kind;
                switch (context.operation.kind) {
                case Kind::DynamicPad:
                case Kind::Cholesky:
                case Kind::SetDimensionSize:
                case Kind::BatchNormGrad:
                case Kind::UnaryEinsum:
                case Kind::Input:
                case Kind::Count:
                    return false;
                default:
                    return true;
                }
            }
    };
}

PUBLISH(llvm_support, Compatibilities::Llvm::Support{})
