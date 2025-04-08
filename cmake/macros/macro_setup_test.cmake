# A small macro used for setting up the build of a test.
#
# Usage: setup_test(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

macro(setup_test namel)
  add_executable(${namel}.${buildl} ${namel}.cpp)

  set_target_properties(${namel}.${buildl}
                        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR}
            ${HARP_INCLUDE_DIR}
            ${DISORT_INCLUDE_DIR}
            ${ELEMENTS_INCLUDE_DIR}
            ${KINTERA_INCLUDE_DIR}
            ${TORCH_INCLUDE_DIR}
            ${TORCH_API_INCLUDE_DIR})

  target_link_libraries(
    ${namel}.${buildl}
    PRIVATE kintera::kintera gtest_main
            $<IF:$<BOOL:${CUDAToolkit_FOUND}>,kintera::kintera_cu,>)

  add_test(NAME ${namel}.${buildl} COMMAND ${namel}.${buildl})
endmacro()
