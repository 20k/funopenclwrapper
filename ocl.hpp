#ifndef OCL_HPP_INCLUDED
#define OCL_HPP_INCLUDED

#include <cl/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

namespace cl
{
    extern std::unordered_map<std::string, std::map<int, const void*>> kernel_map;


    bool supports_extension(cl_device_id device, const std::string& ext_name);

    cl_int get_platform_ids(cl_platform_id* clSelectedPlatformID);

    struct buffer
    {
        cl_mem cmem;
        int64_t alloc_size = 0;

        cl_mem& get()
        {
            return cmem;
        }

        operator cl_mem() {return cmem;}
    };

    struct context
    {
        cl_platform_id platform;
        cl_device_id devices[100] = {0};
        cl_device_id selected_device;
        cl_context ccontext;

        context();

        cl_context& get(){return ccontext;}
    };

    struct program
    {
        cl_program cprogram;
        bool built = false;

        program(context& ctx, const std::string& fname);

        cl_program& get()
        {
            return cprogram;
        }

        cl_program get() const
        {
            return cprogram;
        }

        operator cl_program() {return cprogram;}

        void ensure_build();

        void build_with(context& ctx, const std::string& options);
    };

    struct kernel
    {
        cl_kernel ckernel;
        std::string name;
        bool loaded = false;
        cl_uint work_size;

        kernel(program& p, const std::string& kname);
    };

    struct arg_info
    {
        void* ptr = nullptr;
        int64_t size = 0;
    };

    struct args
    {
        std::vector<arg_info> arg_list;

        template<typename T>
        inline
        void push_back(T& val)
        {
            arg_info inf;
            inf.ptr = &val;
            inf.size = sizeof(T);

            arg_list.push_back(inf);
        }
    };

    struct command_queue
    {
        cl_command_queue cqueue;

        command_queue(context& ctx);

        ///size defaults to -1 which means map the whole buffer
        void* map(buffer& v, cl_map_flags flag, int64_t size = -1);
        void unmap(buffer& v, void* ptr);

        ///make this finally non stupid
        void exec(const std::string& kname, args& pack)
        {

        }
    };

    kernel load_kernel(context& ctx, program& p, const std::string& name);
}

template<>
inline
void cl::args::push_back<cl::buffer>(cl::buffer& val)
{
    cl::arg_info inf;
    inf.ptr = &val.get();
    inf.size = sizeof(val.get());

    arg_list.push_back(inf);
}

#endif // OCL_HPP_INCLUDED
