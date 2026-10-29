#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <windows.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "OpenCL.lib")

#define WARMUP 3
#define RUNS 11
#define CASES 3

static LARGE_INTEGER timer_frequency;

/* ќдинаковое преобразование индекса в слово AAA...ZZZ на CPU и GPU. */
static void make_word(unsigned int index, int length, char word[6])
{
    for (int i = length - 1; i >= 0; --i) {
        word[i] = (char)('A' + index % 26u);
        index /= 26u;
    }
    word[length] = '\0';
}

/* 32-битный FNV-1a дл€ строки известной длины. */
static uint32_t fnv1a(const char* word, int length)
{
    uint32_t hash = 2166136261u;
    for (int i = 0; i < length; ++i) {
        hash ^= (unsigned char)word[i];
        hash *= 16777619u;
    }
    return hash;
}

/* ѕолный последовательный перебор: ищем минимальный индекс совпадени€. */
__declspec(noinline)
static int search_cpu(uint32_t target_hash, int length, unsigned int count)
{
    int found = INT_MAX;
    char word[6];

    for (unsigned int index = 0; index < count; ++index) {
        make_word(index, length, word);
        if (fnv1a(word, length) == target_hash && found == INT_MAX)
            found = (int)index;
    }
    return found;
}

/*  ажда€ задача GPU провер€ет ровно одну комбинацию. */
static const char* gpu_source =
"__kernel void search_fnv(uint target, int length, uint count,\n"
"                         volatile __global int* found) {\n"
"    uint id = (uint)get_global_id(0);\n"
"    if (id >= count) return;\n"
"    uchar word[5];\n"
"    uint value = id;\n"
"    for (int i = length - 1; i >= 0; --i) {\n"
"        word[i] = (uchar)(65u + value % 26u);\n"
"        value /= 26u;\n"
"    }\n"
"    uint hash = 2166136261u;\n"
"    for (int i = 0; i < length; ++i) {\n"
"        hash ^= (uint)word[i];\n"
"        hash *= 16777619u;\n"
"    }\n"
"    if (hash == target) atomic_min(found, (int)id);\n"
"}\n";

static double milliseconds(LARGE_INTEGER begin, LARGE_INTEGER end)
{
    return 1000.0 * (double)(end.QuadPart - begin.QuadPart)
        / (double)timer_frequency.QuadPart;
}

static int compare_doubles(const void* pa, const void* pb)
{
    double a = *(const double*)pa;
    double b = *(const double*)pb;
    return (a > b) - (a < b);
}

static double median(double values[RUNS])
{
    qsort(values, RUNS, sizeof(double), compare_doubles);
    return values[RUNS / 2];
}

/* GPU Total: сброс результата + запуск €дра + чтение результата.
   GPU Kernel: врем€ только вычислительного €дра по OpenCL event. */
static int search_gpu(cl_command_queue queue, cl_kernel kernel,
    cl_mem result_buffer, unsigned int count,
    int* found, double* kernel_time, double* total_time)
{
    const int initial = INT_MAX;
    const size_t global_size = (size_t)count;
    cl_event kernel_event = NULL;
    cl_int err;
    LARGE_INTEGER begin, end;

    QueryPerformanceCounter(&begin);

    err = clEnqueueWriteBuffer(queue, result_buffer, CL_FALSE,
        0, sizeof(initial), &initial, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto failed;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        &global_size, NULL, 0, NULL, &kernel_event);
    if (err != CL_SUCCESS) goto failed;

    err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE,
        0, sizeof(*found), found, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto failed;

    QueryPerformanceCounter(&end);

    if (kernel_time) {
        cl_ulong start_ns = 0, finish_ns = 0;
        err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
            sizeof(start_ns), &start_ns, NULL);
        if (err != CL_SUCCESS) goto failed;
        err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
            sizeof(finish_ns), &finish_ns, NULL);
        if (err != CL_SUCCESS) goto failed;
        *kernel_time = (double)(finish_ns - start_ns) / 1000000.0;
    }
    if (total_time) *total_time = milliseconds(begin, end);

    clReleaseEvent(kernel_event);
    return 1;

failed:
    printf("OpenCL execution error: %d\n", err);
    if (kernel_event) clReleaseEvent(kernel_event);
    clFinish(queue);
    return 0;
}

/* “ри длины слова, 3 прогрева и 11 измерений дл€ каждой длины. */
static int benchmark_case(cl_command_queue queue, cl_kernel kernel,
    cl_mem result_buffer, int length)
{
    unsigned int count = 1u;
    char target_word[6];
    char found_word[6];
    double cpu_times[RUNS], kernel_times[RUNS], total_times[RUNS];
    int cpu_found = INT_MAX, gpu_found = INT_MAX;
    cl_uint hash;
    cl_uint gpu_count;
    cl_int err;

    for (int i = 0; i < length; ++i) count *= 26u;
    make_word(count - 1u, length, target_word);
    hash = (cl_uint)fnv1a(target_word, length);
    gpu_count = (cl_uint)count;

    err = clSetKernelArg(kernel, 0, sizeof(hash), &hash);
    if (err != CL_SUCCESS) goto arg_failed;
    err = clSetKernelArg(kernel, 1, sizeof(length), &length);
    if (err != CL_SUCCESS) goto arg_failed;
    err = clSetKernelArg(kernel, 2, sizeof(gpu_count), &gpu_count);
    if (err != CL_SUCCESS) goto arg_failed;
    err = clSetKernelArg(kernel, 3, sizeof(result_buffer), &result_buffer);
    if (err != CL_SUCCESS) goto arg_failed;

    for (int i = 0; i < WARMUP; ++i) {
        cpu_found = search_cpu(hash, length, count);
        if (!search_gpu(queue, kernel, result_buffer, count,
            &gpu_found, NULL, NULL)) return 0;
    }

    for (int i = 0; i < RUNS; ++i) {
        LARGE_INTEGER begin, end;
        QueryPerformanceCounter(&begin);
        cpu_found = search_cpu(hash, length, count);
        QueryPerformanceCounter(&end);
        cpu_times[i] = milliseconds(begin, end);

        if (!search_gpu(queue, kernel, result_buffer, count,
            &gpu_found, &kernel_times[i], &total_times[i])) return 0;
        if (cpu_found == INT_MAX || cpu_found != gpu_found) {
            printf("Mismatch for length %d: CPU=%d, GPU=%d\n",
                length, cpu_found, gpu_found);
            return 0;
        }
    }

    make_word((unsigned int)cpu_found, length, found_word);
    if (fnv1a(found_word, length) != hash) {
        printf("Hash verification failed for length %d\n", length);
        return 0;
    }
    if (strcmp(found_word, target_word) != 0)
        printf("Note: hash collision for %s, first match is %s\n",
            target_word, found_word);

    printf("%d | %-6s | %10u | %10.6f | %10.6f | %10.6f | OK\n",
        length, target_word, count,
        median(cpu_times), median(kernel_times), median(total_times));
    return 1;

arg_failed:
    printf("OpenCL argument error: %d\n", err);
    return 0;
}

int main(void)
{
    cl_platform_id* platforms = NULL;
    cl_device_id device = NULL;
    cl_uint platform_count = 0;
    cl_int err;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem result_buffer = NULL;
    int passed = 0;
    int exit_code = 1;

    if (!QueryPerformanceFrequency(&timer_frequency)) {
        puts("High-resolution Windows timer unavailable.");
        return 1;
    }

    err = clGetPlatformIDs(0, NULL, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        puts("OpenCL platform not found.");
        return 1;
    }
    platforms = (cl_platform_id*)malloc((size_t)platform_count * sizeof(*platforms));
    if (!platforms) return 1;
    err = clGetPlatformIDs(platform_count, platforms, NULL);
    if (err != CL_SUCCESS) goto cleanup;

    for (cl_uint i = 0; i < platform_count; ++i) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err == CL_SUCCESS) break;
        device = NULL;
    }
    if (!device) {
        puts("OpenCL GPU not found.");
        goto cleanup;
    }

    char device_name[256] = { 0 };
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
        device_name, NULL);
    if (err != CL_SUCCESS) goto cleanup;
    printf("GPU: %s\n", device_name);
    printf("Warmup: %d; measurements per case: %d\n", WARMUP, RUNS);
    puts("Benchmark: full search, first matching hash index");
    puts("Length | Target | Candidates | CPU ms | GPU Kernel ms | GPU Total ms | Check");
    puts("---------------------------------------------------------------------------");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) goto cleanup;
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || !queue) goto cleanup;
    program = clCreateProgramWithSource(context, 1, &gpu_source, NULL, &err);
    if (err != CL_SUCCESS || !program) goto cleanup;
    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192] = { 0 };
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            sizeof(log), log, NULL);
        printf("OpenCL build error %d:\n%s\n", err, log);
        goto cleanup;
    }
    kernel = clCreateKernel(program, "search_fnv", &err);
    if (err != CL_SUCCESS || !kernel) goto cleanup;
    result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS || !result_buffer) goto cleanup;

    for (int length = 3; length <= 5; ++length)
        passed += benchmark_case(queue, kernel, result_buffer, length);

    printf("\nTests passed: %d/%d\n", passed, CASES);
    if (passed == CASES) {
        puts("ALL CHECKS PASSED");
        exit_code = 0;
    }

cleanup:
    if (result_buffer) clReleaseMemObject(result_buffer);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    free(platforms);
    return exit_code;
}
