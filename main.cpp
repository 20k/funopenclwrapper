#include <iostream>
#include <cl/cl.h>
#include "ocl.hpp"
#include "logging.hpp"
#include <SFML/Graphics.hpp>

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    lg::log("Test");

    sf::RenderWindow win;
    win.create(sf::VideoMode(800, 600), "Test");

    cl::context ctx;

    cl::program program(ctx, "test_cl.cl");
    program.build_with(ctx, "");

    //cl::kernel test_kernel(program, "test_kernel");

    cl::command_queue cqueue(ctx);

    cl::args none;

    cqueue.exec(program, "test_kernel", none, {128}, {16});

    cqueue.block();

    return 0;
}
