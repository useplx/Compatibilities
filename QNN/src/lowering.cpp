#include "mln/ir.hpp"
#include "mln/sdk/lowering/compatibility.hpp"
#include "syn/package.hpp"

#include <algorithm>
#include <cstdlib>
#include <print>
#include <unordered_set>
#include <vector>

namespace Compatibilities::Qnn {
    class Lowering final : public Mln::Sdk::Lowering::Compatibility {
        public:
            std::string_view source() const noexcept override { return Mln::Ir::Graph::Format; }
            std::string_view target() const noexcept override { return "qnn-source-v1"; }

            bool applies(const Mln::Sdk::Backend &backend) const override {
                return backend.device == "qnn";
            }

            bool lower(Mln::Ir::Module module, const Context &context) const override {
                if (context.operations.empty())
                    return false;
                std::unordered_set<Mln::Ir::Operation *> selected;
                for (Mln::Ir::Operation *operation : context.operations) {
                    const auto name = operation->getName().getStringRef();
                    const bool contraction =
                        std::string_view(name.data(), name.size()) == Mln::Ir::Graph::DotGeneral;
                    if (contraction) {
                        operation->setAttr(
                            "myelin.lowering",
                            Mln::Ir::StringAttribute::get(operation->getContext(), target()));
                        selected.insert(operation);
                    } else {

                        operation->removeAttr("stream.affinity");
                        operation->removeAttr("myelin.backend");
                    }
                }
                if (selected.empty())
                    return true;
                std::vector<Mln::Ir::Operation *> operations(selected.begin(), selected.end());

                const bool outlined = context.outline && context.outline(module, operations, "qnn");
                if (const char *trace = std::getenv("MLN_TRACE_LOWERING"); trace && *trace)
                    std::println(stderr, "qnn lowering: {} operations selected, outline={}",
                                 operations.size(), outlined);
                return outlined;
            }
    };
}

PUBLISH(qnn_lowering, Compatibilities::Qnn::Lowering{})
