
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "OpenCL.lib")

#define WARMUP 3
#define RUNS 11

static LARGE_INTEGER frequency;
static volatile unsigned long long check_value = 0;

/* OpenCL-ядро транспонирования матрицы */
static const char* gpu_source =
"__kernel void transpose(__global const int* a,\n"
"                        __global int* b,\n"
"                        int rows, int cols) {\n"
"    int j = (int)get_global_id(0);\n"
"    int i = (int)get_global_id(1);\n"
"    if (i < rows && j < cols)\n"
"        b[j * rows + i] = a[i * cols + j];\n"
"}\n";

static double elapsed_ms(LARGE_INTEGER start, LARGE_INTEGER end)
{
    return (double)(end.QuadPart - start.QuadPart) *
        1000.0 / (double)frequency.QuadPart;
}

/* Последовательное транспонирование на CPU */
__declspec(noinline)
static void transpose_cpu(const int* a, int* b, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            b[j * rows + i] = a[i * cols + j];
        }
    }
}

static int compare_double(const void* a, const void* b)
{
    double x = *(const double*)a;
    double y = *(const double*)b;

    return (x > y) - (x < y);
}

static double median(double* values)
{
    qsort(values, RUNS, sizeof(double), compare_double);
    return values[RUNS / 2];
}

/* Запуск GPU и измерение времени OpenCL */
static int execute_gpu(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem input,
    cl_mem output,
    const int* a,
    int* gpu,
    size_t bytes,
    size_t rows,
    size_t cols,
    double* kernel_ms,
    double* total_ms)
{
    cl_int err;
    cl_event event = NULL;
    cl_ulong started = 0;
    cl_ulong ended = 0;

    LARGE_INTEGER t0, t1;
    size_t global[2] = { cols, rows };

    QueryPerformanceCounter(&t0);

    err = clEnqueueWriteBuffer(queue, input,
        CL_FALSE, 0, bytes, a, 0, NULL, NULL);

    if (err != CL_SUCCESS)
        goto failed;

    err = clEnqueueNDRangeKernel(queue, kernel,
        2, NULL, global, NULL, 0, NULL, &event);

    if (err != CL_SUCCESS)
        goto failed;

    err = clEnqueueReadBuffer(queue, output,
        CL_TRUE, 0, bytes, gpu, 0, NULL, NULL);

    if (err != CL_SUCCESS)
        goto failed;

    QueryPerformanceCounter(&t1);

    if (kernel_ms) {
        err = clGetEventProfilingInfo(event,
            CL_PROFILING_COMMAND_START,
            sizeof(started), &started, NULL);

        if (err != CL_SUCCESS)
            goto failed;

        err = clGetEventProfilingInfo(event,
            CL_PROFILING_COMMAND_END,
            sizeof(ended), &ended, NULL);

        if (err != CL_SUCCESS)
            goto failed;

        *kernel_ms = (double)(ended - started) / 1000000.0;
    }

    if (total_ms)
        *total_ms = elapsed_ms(t0, t1);

    clReleaseEvent(event);
    return 1;

failed:
    printf("OpenCL execution error: %d\n", err);

    if (event)
        clReleaseEvent(event);

    clFinish(queue);
    return 0;
}

/* Проверка корректности и измерение одного размера */
static int run_test(
    cl_context context,
    cl_command_queue queue,
    cl_kernel kernel,
    int rows,
    int cols)
{
    size_t count = (size_t)rows * (size_t)cols;
    size_t bytes = count * sizeof(int);

    int* a = (int*)malloc(bytes);
    int* cpu = (int*)malloc(bytes);
    int* gpu = (int*)malloc(bytes);

    cl_mem input = NULL;
    cl_mem output = NULL;
    cl_int err = CL_SUCCESS;

    double cpu_times[RUNS];
    double gpu_kernel_times[RUNS];
    double gpu_total_times[RUNS];

    int success = 0;

    if (!a || !cpu || !gpu) {
        printf("Memory allocation error.\n");
        goto cleanup;
    }

    for (size_t i = 0; i < count; i++)
        a[i] = (int)((i * 37u + 11u) % 100003u);

    input = clCreateBuffer(context,
        CL_MEM_READ_ONLY, bytes, NULL, &err);

    if (err != CL_SUCCESS || !input)
        goto cleanup;

    output = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY, bytes, NULL, &err);

    if (err != CL_SUCCESS || !output)
        goto cleanup;

    err = clSetKernelArg(kernel, 0, sizeof(input), &input);
    if (err != CL_SUCCESS) goto cleanup;

    err = clSetKernelArg(kernel, 1, sizeof(output), &output);
    if (err != CL_SUCCESS) goto cleanup;

    err = clSetKernelArg(kernel, 2, sizeof(rows), &rows);
    if (err != CL_SUCCESS) goto cleanup;

    err = clSetKernelArg(kernel, 3, sizeof(cols), &cols);
    if (err != CL_SUCCESS) goto cleanup;

    /* Предварительные запуски для прогрева */
    for (int i = 0; i < WARMUP; i++) {
        transpose_cpu(a, cpu, rows, cols);

        if (!execute_gpu(queue, kernel, input, output,
            a, gpu, bytes, (size_t)rows, (size_t)cols,
            NULL, NULL))
            goto cleanup;
    }

    /* Повторные измерения CPU и GPU */
    for (int r = 0; r < RUNS; r++) {
        LARGE_INTEGER t0, t1;

        QueryPerformanceCounter(&t0);

        transpose_cpu(a, cpu, rows, cols);

        QueryPerformanceCounter(&t1);

        cpu_times[r] = elapsed_ms(t0, t1);

        check_value += (unsigned int)cpu[(size_t)r % count];

        if (!execute_gpu(queue, kernel, input, output,
            a, gpu, bytes, (size_t)rows, (size_t)cols,
            &gpu_kernel_times[r], &gpu_total_times[r]))
            goto cleanup;

        if (memcmp(cpu, gpu, bytes) != 0) {
            printf("%d x %d: FAILED\n", rows, cols);
            goto cleanup;
        }
    }

    printf("%5d x %-5d | %10.6f | %10.6f | %10.6f | OK\n",
        rows, cols,
        median(cpu_times),
        median(gpu_kernel_times),
        median(gpu_total_times));

    success = 1;

cleanup:
    if (!success && err != CL_SUCCESS)
        printf("OpenCL setup error: %d\n", err);

    if (input)
        clReleaseMemObject(input);

    if (output)
        clReleaseMemObject(output);

    free(a);
    free(cpu);
    free(gpu);

    return success;
}

int main(void)
{
    cl_uint platform_count = 0;
    cl_platform_id* platforms = NULL;
    cl_device_id device = NULL;
    cl_int err;

    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    int result = 1;

    if (!QueryPerformanceFrequency(&frequency)) {
        printf("High-resolution timer unavailable.\n");
        return 1;
    }

    /* Поиск GPU с поддержкой OpenCL */
    err = clGetPlatformIDs(0, NULL, &platform_count);

    if (err != CL_SUCCESS || platform_count == 0) {
        printf("OpenCL platform not found.\n");
        return 1;
    }

    platforms = (cl_platform_id*)malloc(
        platform_count * sizeof(cl_platform_id));

    if (!platforms)
        return 1;

    err = clGetPlatformIDs(platform_count, platforms, NULL);

    if (err != CL_SUCCESS)
        goto cleanup;

    for (cl_uint i = 0; i < platform_count; i++) {
        err = clGetDeviceIDs(platforms[i],
            CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        if (err == CL_SUCCESS)
            break;

        device = NULL;
    }

    if (!device) {
        printf("OpenCL GPU not found.\n");
        goto cleanup;
    }

    char name[256] = { 0 };

    err = clGetDeviceInfo(device, CL_DEVICE_NAME,
        sizeof(name), name, NULL);

    if (err != CL_SUCCESS)
        goto cleanup;

    printf("GPU: %s\n", name);
    printf("Warmup: %d, measurements: %d\n\n", WARMUP, RUNS);

    /* Настройка OpenCL с профилированием GPU */
    context = clCreateContext(NULL, 1, &device,
        NULL, NULL, &err);

    if (err != CL_SUCCESS || !context)
        goto cleanup;

    queue = clCreateCommandQueue(context, device,
        CL_QUEUE_PROFILING_ENABLE, &err);

    if (err != CL_SUCCESS || !queue)
        goto cleanup;

    program = clCreateProgramWithSource(context,
        1, &gpu_source, NULL, &err);

    if (err != CL_SUCCESS || !program)
        goto cleanup;

    err = clBuildProgram(program, 1, &device,
        NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        char log[4096] = { 0 };

        clGetProgramBuildInfo(program, device,
            CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);

        printf("Kernel build error:\n%s\n", log);
        goto cleanup;
    }

    kernel = clCreateKernel(program, "transpose", &err);

    if (err != CL_SUCCESS || !kernel)
        goto cleanup;

    const int sizes[][2] = {
        {2, 3},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2048, 2048},
        {512, 1024}
    };

    int total = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int passed = 0;

    printf("Matrix       | CPU ms     | GPU kernel | GPU total  | Check\n");
    printf("--------------------------------------------------------------\n");

    for (int i = 0; i < total; i++) {
        passed += run_test(context, queue, kernel,
            sizes[i][0], sizes[i][1]);
    }

    printf("--------------------------------------------------------------\n");
    printf("Tests passed: %d/%d\n", passed, total);

    if (passed == total) {
        printf("All tests passed!\n");
        result = 0;
    }
    else {
        printf("Some tests failed.\n");
    }

cleanup:
    if (kernel)
        clReleaseKernel(kernel);

    if (program)
        clReleaseProgram(program);

    if (queue)
        clReleaseCommandQueue(queue);

    if (context)
        clReleaseContext(context);

    free(platforms);

    return result;
}
