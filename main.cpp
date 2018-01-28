#include <iostream>
#include <cl/cl.h>
#include "logging.hpp"

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    lg::log("Test");

    return 0;
}
