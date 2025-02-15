#pragma once

#include <map>
#include <string>
#include <vector>

namespace kintera {

const std::vector<std::string>& element_symbols();
const std::vector<std::string>& element_names();
const std::map<std::string, double>& element_weights();

double get_element_weight(const std::string& ename);
double get_element_weight(int atomicNumber);

std::string get_element_symbol(const std::string& ename);
std::string get_element_symbol(int atomicNumber);
std::string get_element_name(const std::string& ename);
std::string get_element_name(int atomicNumber);

int get_atomic_number(const std::string& ename);
size_t num_elements_defined();
size_t num_isotopes_defined();

}  // namespace kintera
