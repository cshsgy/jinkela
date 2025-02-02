// C/++
#include <sstream>

// torch
#include <torch/torch.h>

// fmt
#include <fmt/format.h>

// kintera
#include "fp_value.hpp"

namespace kintera {

double fp_value(const string& val)
{
    double rval;
    std::stringstream ss(val);
    ss.imbue(std::locale("C"));
    ss >> rval;
    return rval;
}

double fp_value_check(const string& val)
{
    string str = ba::trim_copy(val);
    TORCH_CHECK(!str.empty(), "string has zero length");

    int numDot = 0;
    int numExp = 0;
    char ch;
    int istart = 0;
    ch = str[0];
    if (ch == '+' || ch == '-') {
        TORCH_CHECK(str.size() > 1, fmt::format("string '{}' ends in '{}'", val, ch));
        istart = 1;
    }
    for (size_t i = istart; i < str.size(); i++) {
        ch = str[i];
        if (isdigit(ch)) {
        } else if (ch == '.') {
            numDot++;
            TORCH_CHECK(numDot <= 1, fmt::format("string '{}' has more than one decimal point.", val));
            TORCH_CHECK(numExp == 0, fmt::format("string '{}' has decimal point in exponent", val));
        } else if (ch == 'e' || ch == 'E' || ch == 'd' || ch == 'D') {
            numExp++;
            str[i] = 'E';
            if (numExp > 1) {
                TORCH_CHECK(false, fmt::format("string '{}' has more than one exp char", val));
            } else if (i == str.size() - 1) {
                TORCH_CHECK(false, fmt::format("string '{}' ends in '{}'", val, ch));
            }
            ch = str[i+1];
            if (ch == '+' || ch == '-') {
                TORCH_CHECK(i + 1 < str.size() - 1, fmt::format("string '{}' ends in '{}'", val, ch));
                i++;
            }
        } else {
            TORCH_CHECK(false, fmt::format("Trouble processing string '{}'", str)
        }
    }
    return fp_value(str);
}

} // namespace kintera
