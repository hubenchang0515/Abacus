cmake_minimum_required(VERSION 3.13)
project(AbacusTest)

# 读取源文件列表
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ SOURCES)

# 构建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 引用头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include())

# 链接库
target_link_libraries(${PROJECT_NAME} Abacus)

# 设置编译选项
# 读取源文件列表
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "AMD64")
    set_target_properties(${PROJECT_NAME} 
                            PROPERTIES
                            COMPILE_FLAGS
                            "-std=c++11 -O3 -fopenmp -mavx2")
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm")
    set_target_properties(${PROJECT_NAME} 
                        PROPERTIES
                        COMPILE_FLAGS
                        "-std=c++11 -O3 -fopenmp -mfpu=neon")
endif()

# 设置链接选项
set_target_properties(${PROJECT_NAME} 
                        PROPERTIES
                        LINK_FLAGS 
                        "-fopenmp")