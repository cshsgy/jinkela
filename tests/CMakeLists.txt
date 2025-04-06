# ===============================================
# Examples for testing
# ===============================================

setup_problem(test_yaml_parse)
setup_problem(test_kinetics_rates)
setup_problem(test_eos)

file(GLOB inputs *.inp *.dat *.yaml)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
