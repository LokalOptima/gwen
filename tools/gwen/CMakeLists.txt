set(TARGET gwen)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE gwen_lib)
install(TARGETS ${TARGET} RUNTIME)
