#ifndef OCL_HPP_INCLUDED
#define OCL_HPP_INCLUDED

#include <cl/cl.h>
#include <string>
#include <vector>

namespace ocl
{
    bool supports_extension(cl_device_id device, const std::string& ext_name);

    cl_int get_platform_ids(cl_platform_id* clSelectedPlatformID);

    char* file_contents(const char *filename, int *length);

    struct kernel
    {
        cl_kernel kernel;
        std::string name;
        bool loaded = false;
        cl_uint work_size;
    };

    struct command_queue
    {
        cl_command_queue cqueue;

        void* map(cl_mem& v, cl_map_flags flag, int size);
        void unmap(cl_mem& v, void* ptr);
    };
}

#endif // OCL_HPP_INCLUDED
