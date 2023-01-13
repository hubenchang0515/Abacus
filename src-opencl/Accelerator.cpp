#include "Accelerator.h"

namespace abacus
{

// static 
std::vector<cl::Platform> Accelerator::platforms() noexcept
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    return platforms;
}

// static 
std::vector<cl::Device> Accelerator::devices(cl::Platform platform, cl_device_type type) noexcept
{
    std::vector<cl::Device> devices;
    platform.getDevices(type, &devices);
    return devices;
}

// static 
void Accelerator::printDeviceInfo(cl::Device device) noexcept
{
    std::string name;
    device.getInfo(CL_DEVICE_NAME, &name);
    printf("Name: %s\n", name.c_str());

    size_t value=0;
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &value);
    printf("Compute Units: %zu\n", value);

    device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &value);
    printf("Max Frequency: %zu MHz\n", value);

    device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &value);
    printf("Memory: %zu MiB\n", value / 1024 / 1024);

    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &value);
    printf("Max Group Size: %zu\n", value);
}

Accelerator::Accelerator(cl::Device device, const std::string& source) noexcept:
    m_device{device},
    m_context{device},
    m_queue{m_context},
    m_program{m_context, source}
{
    if (m_program.build() != CL_SUCCESS)
    {
        std::string info;
        m_program.getBuildInfo(m_device, CL_PROGRAM_BUILD_LOG, &info);
        fprintf(stderr, "%s\n", info.c_str());
        return;
    }

    std::vector<cl::Kernel> kernels;
    m_program.createKernels(&kernels);

    std::string name;
    for (auto& kernel : kernels)
    {
        kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &name);
        m_kernels.emplace(name, kernel);
    }
}

cl::Device Accelerator::device() const noexcept
{
    return m_device;
}

cl::Context Accelerator::context() const noexcept
{
    return m_context;
}

cl::CommandQueue Accelerator::queue() const noexcept
{
    return m_queue;
}

cl::Program Accelerator::program() const noexcept
{
    return m_program;
}

cl::Kernel Accelerator::kernel(const std::string& name) const noexcept
{
    return m_kernels.at(name);
}

size_t Accelerator::maxGroupSize() const noexcept
{
    static size_t _maxGroupSize = 0;
    if (_maxGroupSize == 0)
        m_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &_maxGroupSize);
    return _maxGroupSize;
}

cl::NDRange Accelerator::localSize(size_t total) const noexcept
{
    if (total < maxGroupSize())
        return total;
    else
        return maxGroupSize();
}

cl::NDRange Accelerator::globalSize(size_t total) const noexcept
{
    if (total < maxGroupSize())
        return total;
    else
        return (total + maxGroupSize() - 1) / maxGroupSize() * maxGroupSize();
}

}; // namespace abacus
