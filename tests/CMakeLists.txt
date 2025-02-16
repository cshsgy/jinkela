# ===============================================
# Examples for testing
# ===============================================

setup_problem(test_yaml_parse)

file(GLOB inputs *.inp *.dat *.yaml)
foreach(input ${inputs})
    file(COPY ${input} DESTINATION ${CMAKE_BINARY_DIR}/bin)
endforeach()