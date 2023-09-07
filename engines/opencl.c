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


enum memtype {
	MEMTYPE_HOST = 1,
	MEMTYPE_DEVICE,
};

struct opencl_options {
	struct thread_data *td;
	cl_context context;
	cl_mem buffer;
	cl_command_queue command_queue;
	enum memtype memtype;
	void *host_buffer;	/* If memtype == MEMTYPE_HOST */
};

static struct fio_option options[] = {
	{
		.name  = "memtype",
		.lname = "type of memory access, host or device",
		.type  = FIO_OPT_STR,
		.off1  = offsetof(struct opencl_options, memtype),
		.def   = "host",
		.posval = {
			{
				.ival = "host",
				.oval = MEMTYPE_HOST,
				.help = "Use host memory", 
			},
			{
				.ival = "device",
				.oval = MEMTYPE_DEVICE,
				.help = "Use GPU device memory",
			},
		},
	},
	{
		.name	 = NULL,
	},
};

static pthread_mutex_t running_lock = PTHREAD_MUTEX_INITIALIZER;

static unsigned int running = 0;

static cl_platform_id platform_id;
static cl_device_id device_id;


static int init_devices(struct thread_data *td)
{
	cl_int result;

	/* Get the first platform. */
	result = clGetPlatformIDs(1, &platform_id, NULL);
	if (result != CL_SUCCESS) {
		log_err("No platform found: %d", result);
			return 1;
	}

	/* Get the first device of the default type on the platform. */
	result = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,  &device_id, NULL);
	if (result != CL_SUCCESS) {
		log_err("No device found: %d", result);
			return 1;
	}

	return 0;
}

static enum fio_q_status fio_opencl_queue(struct thread_data *td,
					  struct io_u *io_u)
{
	struct opencl_options *o = td->eo;
	cl_int result;
	
	/* Create a command queue */
	if (!o->command_queue) {
		o->command_queue =
			clCreateCommandQueueWithProperties(o->context, device_id, NULL, &result);
		if (result != CL_SUCCESS) {
			log_err("CreateCommandQueueWithProperties failed: %d", result);
			return 1;
		}
	}

	result = clEnqueueWriteBuffer(o->command_queue, o->buffer, CL_TRUE, 0,
				      io_u->buflen, buffer, 0, NULL, NULL);
	if (result != CL_SUCCESS) {
		log_err("CreateCommandQueueWithProperties failed: %d", result);
			return 1;
	}
	return FIO_Q_COMPLETED;
}

static int fio_opencl_init(struct thread_data *td)
 {
	 pthread_mutex_lock(&running_lock);
	 if (!running) {
		 int result = init_devices(td);
		 if (result) {
			 pthread_mutex_unlock(&running_lock);
			 return 1;
		 }
	 } else {
		 
	 }
	 running++;
	 pthread_mutex_unlock(&running_lock);
	 return 0;
 }

static int fio_opencl_iomem_alloc(struct thread_data *td, size_t total_mem)
{
	struct opencl_options *o = td->eo;
	cl_mem_flags flags;
	int result;

	td->orig_buffer = calloc(1, total_mem);
	if (!td->orig_buffer) {
		log_err("Orig Buffer allocation failed\n");
		return 1;
	}
	if (o->memtype == MEMTYPE_HOST) {
		/* Create a memory buffer on the host. */
		flags = CL_MEM_USE_HOST_PTR;
		o->host_buffer = calloc(1, total_mem);
		if (!o->host_buffer) {
			log_err("Host buffer allocation failed");
			return 1;
		}
	} else {
		/* Create a memory buffer on the device. */
		flags = CL_MEM_READ_WRITE;
		o->host_buffer = NULL;
	}

	/* Create a context with default properties. */
	/* TODO: add a callback for error reporting. */
	if (!o->context) {
		o->context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &result);
		if (result != CL_SUCCESS) {
			log_err("CreateContext failed: %d", result);
			return 1;
		}
	}

	o->buffer = clCreateBuffer(o->context, flags, total_mem, o->host_buffer, &result);
	if (result != CL_SUCCESS) {
		log_err("CreateBuffer failed: %d", result);
		return 1;
	}
	return 0;
}

static void fio_opencl_iomem_free(struct thread_data *td)
{
	struct opencl_options *o = td->eo;
	int result;

	result = clReleaseMemObject(o->buffer);
		 
	if (result != CL_SUCCESS) {
		log_err("ReleaseMemObject failed: %d", result);
	}
}

static void fio_opencl_cleanup(struct thread_data *td)
{
	pthread_mutex_lock(&running_lock);
	running--;
	if (!running) {
		/* Tear down */
	}
	pthread_mutex_unlock(&running_lock);
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name                = "OpenCL",
	.version             = FIO_IOOPS_VERSION,
	.init                = fio_opencl_init,
	.queue               = fio_opencl_queue,
	.get_file_size       = generic_get_file_size,
	.open_file           = generic_open_file,
	.close_file          = generic_close_file,
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
