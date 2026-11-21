#include "src/teleop/quest_app/core/json.h"

#include <cstdlib>

namespace qt::json {

const Value* Value::get(std::string_view key) const {
    if (type != Type::Object) return nullptr;
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : &it->second;
}

namespace {

constexpr int kMaxDepth = 32;

struct Parser {
    std::string_view s;
    size_t i = 0;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    bool lit(std::string_view word) {
        if (s.substr(i, word.size()) != word) return false;
        i += word.size();
        return true;
    }

    bool string(std::string& out) {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return true;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (i >= s.size()) return false;
            char e = s[i++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    if (i + 4 > s.size()) return false;
                    unsigned code = std::strtoul(std::string(s.substr(i, 4)).c_str(), nullptr, 16);
                    i += 4;
                    out.push_back(code < 0x80 ? static_cast<char>(code) : '?');
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool number(double& out) {
        size_t start = i;
        if (i < s.size() && s[i] == '-') ++i;
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
                                s[i] == 'E' || s[i] == '+' || s[i] == '-'))
            ++i;
        if (i == start) return false;
        std::string tok(s.substr(start, i - start));
        char* end = nullptr;
        out = std::strtod(tok.c_str(), &end);
        return end == tok.c_str() + tok.size();
    }

    bool value(Value& v, int depth) {
        if (depth > kMaxDepth) return false;
        ws();
        if (i >= s.size()) return false;
        char c = s[i];
        if (c == '{') {
            v.type = Value::Type::Object;
            ++i;
            ws();
            if (i < s.size() && s[i] == '}') return ++i, true;
            while (true) {
                ws();
                std::string key;
                if (!string(key)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return false;
                ++i;
                Value member;
                if (!value(member, depth + 1)) return false;
                v.obj[std::move(key)] = std::move(member);
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') return ++i, true;
                return false;
            }
        }
        if (c == '[') {
            v.type = Value::Type::Array;
            ++i;
            ws();
            if (i < s.size() && s[i] == ']') return ++i, true;
            while (true) {
                Value elem;
                if (!value(elem, depth + 1)) return false;
                v.arr.push_back(std::move(elem));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') return ++i, true;
                return false;
            }
        }
        if (c == '"') {
            v.type = Value::Type::String;
            return string(v.str);
        }
        if (lit("true")) { v.type = Value::Type::Bool; v.b = true; return true; }
        if (lit("false")) { v.type = Value::Type::Bool; v.b = false; return true; }
        if (lit("null")) { v.type = Value::Type::Null; return true; }
        v.type = Value::Type::Number;
        return number(v.num);
    }
};

}  // namespace

std::optional<Value> parse(std::string_view text) {
    Parser p{text};
    Value v;
    if (!p.value(v, 0)) return std::nullopt;
    p.ws();
    if (p.i != text.size()) return std::nullopt;
    return v;
}

}  // namespace qt::json
