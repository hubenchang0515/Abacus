cmake_minimum_required(VERSION 3.10)
project(Abacus)

# 选项
option(BUILD_WITH_OPENCL "build with opencl" OFF)
option(BUILD_WITH_TEST "build with test" OFF)

# 读取源文件列表
if (BUILD_WITH_OPENCL)
    aux_source_directory(${CMAKE_SOURCE_DIR}/src-opencl SOURCES)
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86|X86|amd|AMD)")
    aux_source_directory(${CMAKE_SOURCE_DIR}/src-amd64 SOURCES)
    set(OPT "-mavx2")
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch|AARCH|arm|ARM)")
    aux_source_directory(${CMAKE_SOURCE_DIR}/src-arm64 SOURCES)
    if (NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch64|AARCH64)")
        set(OPT "-mfpu=neon")
    endif()
endif()

# 创建目标
add_library(${PROJECT_NAME} ${SOURCES})

# 引用的头文件目录
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")

# 编译选项
set_target_properties(${PROJECT_NAME} 
                        PROPERTIES
                        COMPILE_FLAGS
                        "-std=c++11 -O3 -fopenmp ${OPT}")


# 读取要安装的头文件列表
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.hpp" "${CMAKE_SOURCE_DIR}/include/*.h")

# 设置要安装的头文件
set_target_properties(${PROJECT_NAME}
                        PROPERTIES
                        PUBLIC_HEADER
                        "${HEADERS}"
)

# 安装
install(TARGETS
        ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

# 单元测试
if (BUILD_WITH_TEST)
    add_subdirectory("test")
endif()
