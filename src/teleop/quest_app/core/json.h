// Minimal JSON reader for the small messages the host sends (RFC 8259 subset: no \u escapes
// beyond ASCII, numbers as double). Writing is done with snprintf in protocol.cpp.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qt::json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value, std::less<>> obj;

    // Object member, or nullptr if absent or this is not an object.
    const Value* get(std::string_view key) const;
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }
};

// nullopt on malformed input or trailing garbage.
std::optional<Value> parse(std::string_view text);

}  // namespace qt::json
