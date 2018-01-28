#include "ocl.hpp"
#include <sstream>
#include "logging.hpp"
#include <cstring>
#include <cl/cl_gl.h>
#include <gl/glew.h>

#include <windows.h>

#include <gl/gl.h>
#include <gl/glext.h>
#include <assert.h>

std::map<std::string, cl::kernel*> cl::kernels;

inline
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

inline
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

bool supports_extension(cl_device_id device, const std::string& ext_name)
{
    size_t rsize;

    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &rsize);

    char* dat = new char[rsize];

    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, rsize, dat, nullptr);

    std::vector<std::string> elements = split(dat, ' ');

    delete [] dat;

    for(auto& i : elements)
    {
        if(i == ext_name)
            return true;
    }

    return false;
}

cl_int cl::get_platform_ids(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;
    cl_uint i = 0;

    ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);

    if(ciErrNum != CL_SUCCESS)
    {
        lg::log("Error ", ciErrNum, " in clGetPlatformIDs");

        return -1000;
    }
    else
    {
        if(num_platforms == 0)
        {
            lg::log("Could not find valid opencl platform, num_platforms == 0");

            return -2000;
        }
        else
        {
            if((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                lg::log("Malloc error for allocating platform ids");

                return -3000;
            }

            ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
            lg::log("Available platforms:");

            for(i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);

                if(ciErrNum == CL_SUCCESS)
                {
                    lg::log("platform ", i, " ", chBuffer);

                    if(strstr(chBuffer, "NVIDIA") != NULL || strstr(chBuffer, "AMD") != NULL)// || strstr(chBuffer, "Intel") != NULL)
                    {
                        lg::log("selected platform ", i);
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            if(*clSelectedPlatformID == NULL)
            {
                lg::log("selected platform ", num_platforms-1);
                *clSelectedPlatformID = clPlatformIDs[num_platforms-1];
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}

std::string read_file(const std::string& file)
{
    FILE *f = fopen(file.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string buffer;
    buffer.resize(fsize + 1);
    fread(&buffer[0], fsize, 1, f);
    fclose(f);

    return buffer;
}

cl::context::context()
{
    kernels.clear();

    cl_int error = 0;   // Used to handle error codes

    error = get_platform_ids(&platform);

    if(error != CL_SUCCESS)
    {
        lg::log("Error getting platform id: ", error);

        exit(error);
    }
    else
    {
        lg::log("Got platform IDs");
    }

    cl_uint num;

    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, devices, &num);

    lg::log("Found ", num, " devices");

    if(error != CL_SUCCESS)
    {
        lg::log("Error getting device ids: ", error);

        exit(error);
    }
    else
    {
        lg::log("Got device ids");
    }

    selected_device = devices[0];

    ///this is essentially black magic
    cl_context_properties props[] =
    {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };

    ccontext = clCreateContext(props, 1, &selected_device, NULL, NULL, &error);

    if(error != CL_SUCCESS)
    {
        lg::log("Error creating context: ", error);

        lg::log("Do you have a valid OpenGL context?");

        exit(error);
    }
    else
    {
        lg::log("Created context");
    }
}

cl::program::program(context& ctx, const std::string& fname)
{
    std::string src = read_file(fname);

    size_t len = src.length();
    const char* ptr = src.c_str();

    cprogram = clCreateProgramWithSource(ctx.get(), 1, &ptr, &len, nullptr);
}

void cl::program::build_with(context& ctx, const std::string& options)
{
    std::string build_options = "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-single-precision-constant -cl-denorms-are-zero" + options;

    cl_int build_status = clBuildProgram(cprogram, 1, &ctx.selected_device, build_options.c_str(), nullptr, nullptr);

    if(build_status != CL_SUCCESS)
    {
        lg::log("Build Error");

        cl_build_status bstatus;
        clGetProgramBuildInfo(cprogram, ctx.selected_device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &bstatus, nullptr);

        assert(bstatus == CL_BUILD_ERROR);

        std::string log;
        size_t log_size;

        clGetProgramBuildInfo(cprogram, ctx.selected_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        log.resize(log_size + 1);

        clGetProgramBuildInfo(cprogram, ctx.selected_device, CL_PROGRAM_BUILD_LOG, log.size(), &log[0], nullptr);

        lg::log(log);

        exit(4);
    }
}

cl::kernel::kernel(program& p, const std::string& kname)
{
    cl_int err;

    ckernel = clCreateKernel(p, kname.c_str(), &err);

    if(err != CL_SUCCESS)
    {
        lg::log("Invalid Kernel Name ", kname);
    }

    name = kname;

    loaded = true;
}

cl::command_queue::command_queue(cl::context& ctx)
{
    cl_int err;

    cqueue = clCreateCommandQueue(ctx.get(), ctx.selected_device, 0, &err);

    if(err != CL_SUCCESS)
    {
        lg::log("Error creating command queue");
    }
}

void* cl::command_queue::map(buffer& v, cl_map_flags flag, int64_t size)
{
    if(size == -1)
        size = v.alloc_size;

    void* ptr = clEnqueueMapBuffer(cqueue, v, CL_TRUE, flag, 0, size, 0, NULL, NULL, NULL);

    if(ptr == nullptr)
    {
        lg::log("error in cl::map");
    }

    return ptr;
}

void cl::command_queue::unmap(buffer& v, void* ptr)
{
    if(ptr == nullptr)
        return;

    clEnqueueUnmapMemObject(cqueue, v, ptr, 0, NULL, NULL);
}

cl::kernel cl::load_kernel(context& ctx, program& p, const std::string& name)
{
    //program_ensure_built();

    kernel k(p, name);

    size_t ret = 128;

    clGetKernelWorkGroupInfo(k.ckernel, ctx.selected_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &ret, NULL);

    k.work_size = ret;

    return k;
}
