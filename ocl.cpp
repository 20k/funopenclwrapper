#include "ocl.hpp"
#include <sstream>
#include "logging.hpp"
#include <cstring>

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

cl_int get_platform_ids(cl_platform_id* clSelectedPlatformID)
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

char* file_contents(const char *filename, int *length)
{
    FILE *f = fopen(filename, "r");
    void *buffer;

    if(!f)
    {
        lg::log("Unable to open ", filename, " for reading");
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = malloc(*length+1);
    *length = fread(buffer, 1, *length, f);
    fclose(f);
    ((char*)buffer)[*length] = '\0';

    return (char*)buffer;
}

void* ocl::command_queue::map(cl_mem& v, cl_map_flags flag, int size)
{
    void* ptr = clEnqueueMapBuffer(cqueue, v, CL_TRUE, flag, 0, size, 0, NULL, NULL, NULL);

    if(ptr == nullptr)
    {
        lg::log("error in cl::map");
    }

    return ptr;
}

void ocl::command_queue::unmap(cl_mem& v, void* ptr)
{
    clEnqueueUnmapMemObject(cqueue, v, ptr, 0, NULL, NULL);
}
