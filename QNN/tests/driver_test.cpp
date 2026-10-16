#include "mln/sdk/driver/module.hpp"

#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    std::vector<std::uint8_t> read(const std::filesystem::path &path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
            throw std::runtime_error("failed to open " + path.string());
        const auto size = stream.tellg();
        stream.seekg(0);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        stream.read(reinterpret_cast<char *>(bytes.data()), size);
        return bytes;
    }
}

int main(int argc, char **argv) {
    if (argc != 5)
        throw std::invalid_argument("driver_test DRIVER CONTEXT INPUT EXPECTED");
    void *library = ::dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
        throw std::runtime_error(::dlerror());
    using Entry = const Mln::Sdk::Driver::Module *(*)();
    const auto entry = reinterpret_cast<Entry>(::dlsym(library, "mln_runtime_driver"));
    if (!entry)
        throw std::runtime_error("mln_runtime_driver is missing");
    const Mln::Sdk::Driver::Module &module = *entry();
    if (module.abi_version != Mln::Sdk::Driver::AbiVersion ||
        module.struct_size < sizeof(Mln::Sdk::Driver::Module))
        throw std::runtime_error("driver ABI is incompatible");

    const std::vector<std::uint8_t> context = read(argv[2]);
    std::vector<std::uint8_t> input = read(argv[3]);
    const std::vector<std::uint8_t> expected = read(argv[4]);
    std::vector<std::uint8_t> output(expected.size());
    char error[1024]{};
    void *executable = module.load(context.data(), context.size(), error, sizeof(error));
    if (!executable)
        throw std::runtime_error(error);
    const std::size_t count = module.function_count(executable);
    if (count != 1)
        throw std::runtime_error("expected one QNN graph");
    Mln::Sdk::Driver::Function function{};
    if (module.function(executable, 0, &function, error, sizeof(error)) != 0)
        throw std::runtime_error(error);
    const Mln::Sdk::Driver::Binding bindings[]{
        {input.data(), input.size()},
        {output.data(), output.size()},
    };
    if (module.invoke(executable, 0, nullptr, 0, bindings, 2, error, sizeof(error)) != 0)
        throw std::runtime_error(error);
    module.destroy(executable);
    ::dlclose(library);

    if (output != expected) {
        const std::size_t differences = std::ranges::count_if(
            std::views::iota(std::size_t{0}, output.size()),
            [&](std::size_t index) { return output[index] != expected[index]; });
        throw std::runtime_error("QNN output differs in " + std::to_string(differences) + " bytes");
    }
    std::cout << function.name << ": " << output.size() << " bytes verified\n";
}
