// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright 2023 Red Hat
 *
 * OpenCL engine
 *
 * fio I/O engine using OpenCL.
 *
 */

#include <CL/cl.h>
#include <pthread.h>
#include <stdbool.h>

#include "../fio.h"

#define BUFFER_SIZE (1024 * 1024)

static char buffer[BUFFER_SIZE] = { 0 };

struct opencl_options {
	struct thread_data *td;
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_context context;
	cl_mem buffer;
	cl_command_queue command_queue;
};

static pthread_mutex_t running_lock = PTHREAD_MUTEX_INITIALIZER;

static unsigned int running = 0;

static int init_devices(struct thread_data *td);
{
	struct opencl_options *o = td->eo;
	cl_uint result;

	/* Get the first platform. */
	result = clGetPlatformIDs(1, &o->platform_id, NULL);
	if (result != CL_SUCCESS) {
		log_err("No platform found: %d", result)
			return 1;
	}

	/* Get the first device of the default type on the platform. */
	result = clGetDeviceIDs(o->platform_id, CL_DEVICE_TYPE_DEFAULT, 1,  &o->device_id, NULL);
	if (result != CL_SUCCESS) {
		log_err("No device found: %d", result)
			return 1;
	}

	/* Create a context with default properties. */
	/* TODO: add a callback for error reporting. */
	o->context = clCreateContext( NULL, 1, &o->device_id, NULL, NULL, &result);
	if (result != CL_SUCCESS) {
		log_err("CreateContext failed: %d", result)
			return 1;
	}

	/* Create a memory buffer on the device. */
	o->buffer = clCreateBuffer(o->context, CL_MEM_READ_WRITE, BUFFER_SIZE, NULL, &result);
	if (result != CL_SUCCESS) {
		log_err("CreateBuffer failed: %d", result)
			return 1;
	}

	/* Create a command queue */
	o->command_queue =
		clCreateCommandQueueWithProperties(o->context, o->device_id, NULL, &result);
	if (result != CL_SUCCESS) {
		log_err("CreateCommandQueueWithProperties failed: %d", result)
			return 1;
	}
	result = clEnqueueWriteBuffer(o->command_queue, o->buffer, CL_TRUE, 0,
				      sizeof(buffer), buffer, 0, NULL, NULL);
	if (result != CL_SUCCESS) {
		log_err("CreateCommandQueueWithProperties failed: %d", result)
			return 1;
	}
	return 0;
}

static int fio_opencl_init(struct thread_data *td))
 {
	 pthread_mutex_lock(&running_lock);
	 if (!running) {
		 int result = init_devices();
		 if (!result) {
			 pthread_mutex_unlock(&running_lock);
			 return result;
		 }
	 }
	 running++;
	 pthread_mutex_unlock(&running_lock);
	 return result;
 }

static void fio_opencl_cleanup(struct thread_data *td)
{
	pthead_mutex_lock(&running_lock);
	running--;
	if (!running) {
		/* Tear down */
	}
	pthead_mutex_lock(&running_unlock);
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name                = "OpenCL",
	.version             = FIO_IOOPS_VERSION,
	.init                = fio_opencl_init,
	.queue               = fio_opencl_queue,
	.get_file_size       = generic_get_file_size,
	.open_file           = fio_opencl_open_file,
	.close_file          = fio_opencl_close_file,
	.iomem_alloc         = fio_opencl_iomem_alloc,
	.iomem_free          = fio_opencl_iomem_free,
	.cleanup             = fio_opencl_cleanup,
	.flags               = FIO_SYNCIO,
	.options             = options,
	.option_struct_size  = sizeof(struct opencl_options)
};

void fio_init fio_opencl_register(void)
{
	register_ioengine(&ioengine);
}

void fio_exit fio_opencl_unregister(void)
{
	unregister_ioengine(&ioengine);
}
