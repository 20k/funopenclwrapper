#ifndef OCL_HPP_INCLUDED
#define OCL_HPP_INCLUDED

#include <cl/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include "logging.hpp"

namespace cl
{
    struct kernel;
    extern std::map<std::string, kernel*> kernels;

    bool supports_extension(cl_device_id device, const std::string& ext_name);

    cl_int get_platform_ids(cl_platform_id* clSelectedPlatformID);

    struct context
    {
        cl_platform_id platform;
        cl_device_id devices[100] = {0};
        cl_device_id selected_device;
        cl_context ccontext;

        context();

        void rebuild();

        cl_context& get(){return ccontext;}

        operator cl_context() {return ccontext;}
    };

    struct program
    {
        context& saved_context;
        std::string saved_fname;

        cl_program cprogram;
        bool built = false;
        program(context& ctx, const std::string& fname);

        void rebuild();

        cl_program& get()
        {
            return cprogram;
        }

        cl_program get() const
        {
            return cprogram;
        }

        void operator=(const program& other)
        {
            cprogram = other.cprogram;
            built = other.built;
            saved_fname = other.saved_fname;
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
        //cl_uint work_size;

        kernel(program& p, const std::string& kname);

        cl_kernel& get(){return ckernel;}
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

    struct buffer;

    template<typename T>
    struct map_info
    {
        buffer& v;
        T* ptr = nullptr;

        map_info(T* ptr, buffer& v) : ptr(ptr), v(v)
        {

        }
    };

    struct command_queue
    {
        cl_command_queue cqueue;

        command_queue(context& ctx);

        ///size defaults to -1 which means map the whole buffer
        void* map(buffer& v, cl_map_flags flag, int64_t size = -1);
        void unmap(buffer& v, void* ptr);

        template<typename T>
        map_info<T> map_type(buffer& v, cl_map_flags flag, int64_t size = -1)
        {
            void* ptr = map(v, flag, size);

            map_info<T> ret(ptr, v);

            return ret;
        }

        template<typename T>
        void unmap(map_info<T>& info)
        {
            unmap(info.v, info.ptr);
        }

        ///make this finally non stupid
        template<typename T, int dim>
        void exec(kernel& kname, args& pack, const T(&global_ws)[dim], const T(&local_ws)[dim])
        {
            for(int i=0; i < (int)pack.arg_list.size(); i++)
            {
                clSetKernelArg(kname.ckernel, i, pack.arg_list[i].size, pack.arg_list[i].ptr);
            }

            size_t g_ws[dim] = {0};
            size_t l_ws[dim] = {0};

            for(int i=0; i < dim; i++)
            {
                l_ws[i] = local_ws[i];

                if(l_ws[i] == 0)
                    continue;

                if((g_ws[i] % l_ws[i]) != 0)
                {
                    int rem = g_ws[i] % l_ws[i];

                    g_ws[i] -= rem;
                    g_ws[i] += l_ws[i];
                }

                if(g_ws[i] == 0)
                {
                    g_ws[i] += l_ws[i];
                }
            }

            cl_int err = clEnqueueNDRangeKernel(cqueue, kname.get(), dim, nullptr, g_ws, l_ws, 0, nullptr, nullptr);

            if(err != CL_SUCCESS)
            {
                lg::log("clEnqueueNDRangeKernel Error with", kname.name);
                lg::log(err);
            }
        }

        template<typename T, int dim>
        void exec(program& p, const std::string& kname, args& pack, const T(&global_ws)[dim], const T(&local_ws)[dim])
        {
            kernel*& k = kernels[kname];

            if(k == nullptr || !k->loaded)
            {
                k = new kernel(p, kname);
            }

            return exec(*k, pack, global_ws, local_ws);
        }

        void block()
        {
            clFinish(cqueue);
        }

        operator cl_command_queue() {return cqueue;}
    };

    ///need a centralised way to invalidate all buffers
    ///associated with a context
    ///and then reallocate them
    struct buffer
    {
        cl_mem cmem;
        int64_t alloc_size = 0;
        context& ctx;

        buffer(context& ctx) : ctx(ctx) {}

        cl_mem& get()
        {
            return cmem;
        }

        void write_all(command_queue& write_on, const void* ptr)
        {
            clEnqueueWriteBuffer(write_on, cmem, CL_TRUE, 0, alloc_size, ptr, 0, nullptr, nullptr);
        }

        template<typename T>
        void write_all(command_queue& write_on, std::vector<T>& data)
        {
            if(data.size() == 0)
                return;

            clEnqueueWriteBuffer(write_on, cmem, CL_TRUE, 0, data.size() * sizeof(T), &data[0], 0, nullptr, nullptr);
        }

        template<typename T>
        std::vector<T> read_all(command_queue& read_on)
        {
            std::vector<T> ret;

            if(alloc_size == 0)
                return ret;

            ret.resize(alloc_size / sizeof(T));

            clEnqueueReadBuffer(read_on, cmem, CL_TRUE, 0, alloc_size, &ret[0], 0, nullptr, nullptr);

            return ret;
        }

        template<typename T>
        void alloc_n(command_queue& write_on, const T* data, int num)
        {
            cl_int err;

            alloc_size = sizeof(T) * num;

            cmem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, alloc_size, nullptr, &err);

            if(err != CL_SUCCESS)
            {
                lg::log("Error allocating buffer");
                return;
            }

            write_all(write_on, data);
        }

        template<typename T>
        void alloc(command_queue& write_on, const std::vector<T>& data)
        {
            if(data.size() == 0)
                return;

            alloc_n(write_on, &data[0], data.size());
        }

        void release()
        {
            clReleaseMemObject(cmem);
        }

        operator cl_mem() {return cmem;}
    };

    struct buffer_manager
    {
        std::map<buffer*, buffer*> buffers;

        buffer* fetch(context& ctx, buffer* old)
        {
            if(old == nullptr)
            {
                buffer* buf = new buffer(ctx);

                buffers[buf] = buf;

                return buf;
            }
            else
            {
                return buffers[old];
            }
        }
    };

    //kernel load_kernel(context& ctx, program& p, const std::string& name);
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

template<>
inline
void cl::args::push_back<cl::buffer*>(cl::buffer*& val)
{
    cl::arg_info inf;
    inf.ptr = &val->get();
    inf.size = sizeof(val->get());

    arg_list.push_back(inf);
}

#endif // OCL_HPP_INCLUDED
