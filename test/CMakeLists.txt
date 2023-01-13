cmake_minimum_required(VERSION 3.10)
project(AbacusTest)

# 读取源文件列表
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ TEST_SOURCE)

# 构建可执行文件
add_executable(${PROJECT_NAME} ${TEST_SOURCE})

# 引用头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# 链接库
target_link_libraries(${PROJECT_NAME} Abacus)

# 设置编译选项
# 读取源文件列表
if (BUILD_WITH_OPENCL)
    set_target_properties(${PROJECT_NAME} 
                            PROPERTIES
                            COMPILE_FLAGS
                            "-std=c++11 -O3")
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86|X86|amd|AMD)")
    set_target_properties(${PROJECT_NAME} 
                            PROPERTIES
                            COMPILE_FLAGS
                            "-std=c++11 -O3 -fopenmp -mavx2")
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch64|AARCH64)")
    set_target_properties(${PROJECT_NAME} 
                        PROPERTIES
                        COMPILE_FLAGS
                        "-std=c++11 -O3 -fopenmp")
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch|AARCH|arm|ARM)")
    set_target_properties(${PROJECT_NAME} 
                        PROPERTIES
                        COMPILE_FLAGS
                        "-std=c++11 -O3 -fopenmp -mfpu=neon")
endif()

# 设置链接选项
if (BUILD_WITH_OPENCL)
    target_link_libraries(${PROJECT_NAME} Abacus OpenCL)
else()
    target_link_libraries(${PROJECT_NAME} Abacus)
    set_target_properties(${PROJECT_NAME} 
                            PROPERTIES
                            LINK_OPTIONS 
                            "-fopenmp")
endif()

