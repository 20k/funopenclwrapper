__kernel
void test_kernel(__global int* test)
{
    int id = get_global_id(0);

    printf("%i ", test[id]);
}
