#include "blackforge/ir/value.hpp"

#include <sstream>

namespace blackforge::ir {

std::string Dim::toString() const {
    if (isSymbolic) {
        return symbolicName;
    }
    return std::to_string(literalValue);
}

std::string typeString(const Value& value) {
    std::ostringstream out;
    out << sema::dtypeName(value.dtype) << "[";
    for (std::size_t i = 0; i < value.shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << value.shape[i].toString();
    }
    out << "]";
    return out.str();
}

}  // namespace blackforge::ir
