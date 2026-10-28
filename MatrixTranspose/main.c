
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <string.h>

#pragma comment(lib, "OpenCL.lib")

#define ROWS 2
#define COLS 3

#define CHECK(call) do { \
    err = (call); \
    if (err != CL_SUCCESS) { \
        printf("OpenCL error %d at line %d\n", err, __LINE__); \
        goto cleanup; \
    } \
} while (0)

/* Функция транспонирования, выполняемая потоками GPU.
   Каждый поток переносит один элемент матрицы. */
static const char* gpu_source =
"__kernel void transpose(__global const int* a,\n"
"                        __global int* b,\n"
"                        int rows, int cols) {\n"
"    int j = (int)get_global_id(0);\n"
"    int i = (int)get_global_id(1);\n"
"    if (i < rows && j < cols)\n"
"        b[j * rows + i] = a[i * cols + j];\n"
"}\n";

int main(void)
{
    int a[ROWS * COLS] = { 1, 2, 3, 4, 5, 6 };
    int cpu[ROWS * COLS] = { 0 };
    int gpu[ROWS * COLS] = { 0 };

    cl_platform_id platforms[8];
    cl_device_id device = NULL;
    cl_uint platform_count = 0;
    cl_int err = CL_SUCCESS;

    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem input_buffer = NULL;
    cl_mem output_buffer = NULL;

    int result = 1;

    /* Транспонирование на CPU.
       Строки исходной матрицы становятся столбцами. */
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            cpu[j * ROWS + i] = a[i * COLS + j];
        }
    }

    /* Поиск доступной платформы OpenCL и видеокарты. */
    CHECK(clGetPlatformIDs(8, platforms, &platform_count));

    if (platform_count > 8)
        platform_count = 8;

    for (cl_uint p = 0; p < platform_count; p++) {
        err = clGetDeviceIDs(platforms[p],
            CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        if (err == CL_SUCCESS)
            break;

        device = NULL;
    }

    if (device == NULL) {
        printf("OpenCL GPU not found.\n");
        goto cleanup;
    }

    char device_name[256] = { 0 };

    CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME,
        sizeof(device_name), device_name, NULL));

    printf("GPU device: %s\n\n", device_name);

    /* Создание контекста и очереди команд для GPU. */
    context = clCreateContext(NULL, 1, &device,
        NULL, NULL, &err);

    if (err != CL_SUCCESS || context == NULL)
        goto cleanup;

    queue = clCreateCommandQueue(context, device, 0, &err);

    if (err != CL_SUCCESS || queue == NULL)
        goto cleanup;

    /* Компиляция программы транспонирования для GPU. */
    program = clCreateProgramWithSource(context, 1,
        &gpu_source, NULL, &err);

    if (err != CL_SUCCESS || program == NULL)
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

    if (err != CL_SUCCESS || kernel == NULL)
        goto cleanup;

    /* Выделение памяти GPU и передача исходной матрицы. */
    input_buffer = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(a), a, &err);

    if (err != CL_SUCCESS || input_buffer == NULL)
        goto cleanup;

    output_buffer = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY, sizeof(gpu), NULL, &err);

    if (err != CL_SUCCESS || output_buffer == NULL)
        goto cleanup;

    int rows = ROWS;
    int cols = COLS;

    CHECK(clSetKernelArg(kernel, 0,
        sizeof(input_buffer), &input_buffer));

    CHECK(clSetKernelArg(kernel, 1,
        sizeof(output_buffer), &output_buffer));

    CHECK(clSetKernelArg(kernel, 2,
        sizeof(rows), &rows));

    CHECK(clSetKernelArg(kernel, 3,
        sizeof(cols), &cols));

    /* Запуск параллельного транспонирования на GPU. */
    size_t global[2] = { COLS, ROWS };

    CHECK(clEnqueueNDRangeKernel(queue, kernel,
        2, NULL, global, NULL, 0, NULL, NULL));

    /* Получение результата вычислений с GPU. */
    CHECK(clEnqueueReadBuffer(queue, output_buffer,
        CL_TRUE, 0, sizeof(gpu), gpu,
        0, NULL, NULL));

    printf("CPU result:\n");

    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++)
            printf("%d ", cpu[i * ROWS + j]);

        printf("\n");
    }

    printf("\nGPU result:\n");

    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++)
            printf("%d ", gpu[i * ROWS + j]);

        printf("\n");
    }

    /* Проверка совпадения результатов CPU и GPU. */
    if (memcmp(cpu, gpu, sizeof(cpu)) == 0) {
        printf("\nCPU and GPU results match!\n");
        result = 0;
    }
    else {
        printf("\nERROR: Results differ!\n");
    }

cleanup:
    /* Освобождение ресурсов OpenCL. */
    if (output_buffer) clReleaseMemObject(output_buffer);
    if (input_buffer) clReleaseMemObject(input_buffer);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);

    return result;
}
