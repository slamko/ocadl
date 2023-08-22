#define CL_TARGET_OPENCL_VERSION 220

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define VEC_LEN 0x40000
#define VEC_SIZE (VEC_LEN * sizeof(int))

int main() {
    int *a = malloc(VEC_SIZE);
    int *b = malloc(VEC_SIZE);

    for (int i = 0; i < VEC_LEN; i++) {
        a[i] = i;
        b[i] = VEC_LEN - i;
    }

    FILE *fp = fopen ("add.c", "r");

    if (!fp) {
        fprintf(stderr, "No source file found");
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    size_t src_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *src = malloc(src_len + 1);
    if (fread(src, sizeof *src, src_len, fp) != src_len) {
        fprintf(stderr, "Read failed");
        exit(1);
    }
    fclose(fp);

    cl_platform_id platform_id = {0};
    cl_device_id device_id = {0};   
    cl_uint ret_num_devices = {0};
    cl_uint ret_num_platforms = {0};
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    if (ret) {
        fprintf(stderr, "No CL platforms found\n");
        exit(1);
    }

    printf("CL platforms number: %d\n", ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, 
            &device_id, &ret_num_devices);

    if (ret) {
        fprintf(stderr, "No CL devices found %d\n", ret);
        exit(1);
    }
 
    // Create an OpenCL context
    cl_context context =
        clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    if (ret) {
        fprintf(stderr, "Failed to instantiate the context, error = %d\n",
                ret);
        exit(1);
    }
 
    // Create a command queue
    cl_command_queue command_queue =
        clCreateCommandQueue(context, device_id, 0, &ret);

    if (ret) {
        fprintf(stderr, "Failed to instantiate the command queue\n");
        exit(1);
    }

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            VEC_SIZE, NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VEC_SIZE, NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            VEC_SIZE, NULL, &ret);
 // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            VEC_LEN * sizeof(int), a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            VEC_LEN * sizeof(int), b, 0, NULL, NULL);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&src, (const size_t *)&src_len, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = VEC_LEN;
    size_t local_item_size  = 256; 

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);

    if (ret) {
        printf("Kernel execution failed\n");
        exit(1);
    }
 
    // Read the memory buffer C on the device to the local variable C
    int *c = malloc(VEC_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            VEC_SIZE, c, 0, NULL, NULL);
 
    /* for(int i = 0; i < VEC_LEN; i++) */
        /* printf("%d + %d = %d\n", a[i], b[i], c[i]); */
 
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(a);
    free(b);
    free(c);
    
    return 0;
}
