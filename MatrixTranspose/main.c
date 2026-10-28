
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "OpenCL.lib")

/* Транспонирование матрицы на GPU */
static const char* gpu_source =
"__kernel void transpose(__global const int* a,\n"
"                        __global int* b,\n"
"                        int rows, int cols) {\n"
"    int j = (int)get_global_id(0);\n"
"    int i = (int)get_global_id(1);\n"
"    if (i < rows && j < cols)\n"
"        b[j * rows + i] = a[i * cols + j];\n"
"}\n";

/* Проверка одного размера матрицы */
static int run_test(cl_context context,
    cl_command_queue queue,
    cl_kernel kernel,
    int rows, int cols)
{
    size_t count = (size_t)rows * cols;
    size_t bytes = count * sizeof(int);

    int* a = (int*)malloc(bytes);
    int* cpu = (int*)malloc(bytes);
    int* gpu = (int*)malloc(bytes);

    cl_mem input = NULL;
    cl_mem output = NULL;
    cl_int err = CL_SUCCESS;
    int success = 0;

    if (!a || !cpu || !gpu) {
        printf("Memory allocation failed.\n");
        goto cleanup;
    }

    /* Заполнение исходной матрицы */
    for (size_t k = 0; k < count; k++)
        a[k] = (int)((k * 37u + 11u) % 100003u);

    /* Последовательное транспонирование на CPU */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cpu[j * rows + i] = a[i * cols + j];
        }
    }

    /* Подготовка памяти GPU */
    input = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, a, &err);

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

    /* Параллельное транспонирование на GPU */
    size_t global[2] = {
        (size_t)cols,
        (size_t)rows
    };

    err = clEnqueueNDRangeKernel(queue, kernel,
        2, NULL, global, NULL, 0, NULL, NULL);

    if (err != CL_SUCCESS) goto cleanup;

    err = clEnqueueReadBuffer(queue, output,
        CL_TRUE, 0, bytes, gpu, 0, NULL, NULL);

    if (err != CL_SUCCESS) goto cleanup;

    /* Сравнение результатов CPU и GPU */
    if (memcmp(cpu, gpu, bytes) == 0) {
        printf("Matrix %d x %d: OK\n", rows, cols);
        success = 1;
    }
    else {
        printf("Matrix %d x %d: FAILED\n", rows, cols);
    }

cleanup:
    if (!success && err != CL_SUCCESS)
        printf("OpenCL error: %d\n", err);

    if (input) clReleaseMemObject(input);
    if (output) clReleaseMemObject(output);

    free(a);
    free(cpu);
    free(gpu);

    return success;
}

int main(void)
{
    cl_platform_id platforms[8];
    cl_platform_id selected_platform = NULL;
    cl_device_id device = NULL;
    cl_uint platform_count = 0;
    cl_int err;

    /* Поиск видеокарты с поддержкой OpenCL */
    err = clGetPlatformIDs(8, platforms, &platform_count);

    if (err != CL_SUCCESS || platform_count == 0) {
        printf("OpenCL platform not found.\n");
        return 1;
    }

    if (platform_count > 8)
        platform_count = 8;

    for (cl_uint i = 0; i < platform_count; i++) {
        err = clGetDeviceIDs(platforms[i],
            CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        if (err == CL_SUCCESS) {
            selected_platform = platforms[i];
            break;
        }
    }

    if (!device || !selected_platform) {
        printf("GPU not found.\n");
        return 1;
    }

    char name[256] = { 0 };

    err = clGetDeviceInfo(device, CL_DEVICE_NAME,
        sizeof(name), name, NULL);

    if (err != CL_SUCCESS)
        return 1;

    printf("GPU: %s\n\n", name);

    /* Создание окружения OpenCL */
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    int result = 1;

    context = clCreateContext(NULL, 1, &device,
        NULL, NULL, &err);

    if (err != CL_SUCCESS || !context)
        goto cleanup;

    queue = clCreateCommandQueue(context, device, 0, &err);

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
            CL_PROGRAM_BUILD_LOG,
            sizeof(log), log, NULL);

        printf("Build error:\n%s\n", log);
        goto cleanup;
    }

    kernel = clCreateKernel(program, "transpose", &err);

    if (err != CL_SUCCESS || !kernel)
        goto cleanup;

    /* Размеры матриц для тестирования */
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

    for (int i = 0; i < total; i++) {
        passed += run_test(context, queue, kernel,
            sizes[i][0], sizes[i][1]);
    }

    printf("\nTests passed: %d/%d\n", passed, total);

    if (passed == total) {
        printf("All tests passed!\n");
        result = 0;
    }
    else {
        printf("Some tests failed.\n");
    }

cleanup:
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);

    return result;
}
