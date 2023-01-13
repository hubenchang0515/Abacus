#ifndef ABACUS_ACCELERATOR_H
#define ABACUS_ACCELERATOR_H

#include <map>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

namespace abacus
{

class Accelerator
{
public:    
    static std::vector<cl::Platform> platforms() noexcept;
    static std::vector<cl::Device> devices(cl::Platform platform, cl_device_type type=CL_DEVICE_TYPE_ALL) noexcept;
    static void printDeviceInfo(cl::Device device) noexcept;

    ~Accelerator() noexcept = default;
    Accelerator(const Accelerator&) noexcept = default;
    Accelerator(Accelerator&&) noexcept = default;
    Accelerator(cl::Device device, const std::string& source) noexcept;

    cl::Device device() const noexcept;
    cl::Context context() const noexcept;
    cl::CommandQueue queue() const noexcept;
    cl::Program program() const noexcept;
    cl::Kernel kernel(const std::string& name) const noexcept;

    size_t maxGroupSize() const noexcept;
    cl::NDRange localSize(size_t total) const noexcept;
    cl::NDRange globalSize(size_t total) const noexcept;

private:
    cl::Device m_device;
    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::Program m_program;
    std::map<std::string, cl::Kernel> m_kernels;

}; // class Accelerator

}; // namespace abacus

#endif // ABACUS_ACCELERATOR_H