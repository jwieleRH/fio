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
#include "../optgroup.h"

#define BUFFER_SIZE (1024 * 1024)

static char buffer[BUFFER_SIZE] = { 0 };


enum memtype {
	MEMTYPE_HOST = 1,
	MEMTYPE_DEVICE,
};

struct opencl_options {
	struct thread_data *td;
	int platform;
	int device;
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
		.category = FIO_OPT_C_ENGINE,
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
		.name    = "platform",
		.lname   = "platform number",
		.type    = FIO_OPT_INT,
		.off1    = offsetof(struct opencl_options, platform),
		.help    = "Platform number, default 0. See clinfo(1).",
		.def     = 0,
		.category = FIO_OPT_C_ENGINE,
	},
	{
		.name    = "device",
		.lname   = "device number",
		.type    = FIO_OPT_INT,
		.off1    = offsetof(struct opencl_options, device),
		.help    = "Device number on the platform, default 0",
		.def     = 0,
		.category = FIO_OPT_C_ENGINE,
	},
	{
		.name	 = NULL,
	},
};

static pthread_mutex_t running_lock = PTHREAD_MUTEX_INITIALIZER;

static unsigned int running = 0;

static cl_platform_id platform_id;
static cl_device_id device_id;

static cl_int print_platform_info(cl_platform_id platform_id)
{
	static char platform_name[128]; /* Is there a known max size? */
	cl_platform_info param_name;
	size_t param_value_size;
	void *param_value;
	size_t param_value_size_ret;
	cl_int result;

	param_name = CL_PLATFORM_NAME;
	param_value_size = sizeof(platform_name);
	param_value = platform_name;

	result = clGetPlatformInfo(platform_id, param_name, param_value_size, param_value,
				   &param_value_size_ret);
				   
	if (result != CL_SUCCESS) {
		log_err("getPlatformInfo failed: %d\n", result);
		return 1;
	}
	log_info("  Platform name: %s\n", platform_name);
	return 0;
}

static cl_int print_device_info(cl_device_id device_id)
{
	static char device_name[128];
	cl_device_info param_name;
	size_t param_value_size;
	void *param_value;
	size_t param_value_size_ret;
	cl_int result;

	param_name = CL_DEVICE_NAME;
	param_value_size = sizeof(device_name);
	param_value = device_name;
	
	result = clGetDeviceInfo(device_id, param_name, param_value_size, param_value,
				 &param_value_size_ret);
	if (result != CL_SUCCESS) {
		return 1;
	}
	log_info("  Device Name:%s\n", device_name);
	return 0;
}

static int init_devices(struct thread_data *td){
	struct opencl_options *o = td->eo;
	cl_platform_id *platforms;
	cl_uint num_platforms;
	cl_device_id *devices;
	cl_uint num_devices;
	cl_int result;

	/* Get the count of available platforms. */
	result = clGetPlatformIDs(0, NULL, &num_platforms);
	if (result != CL_SUCCESS || !num_platforms) {
		log_err("No platforms found: %d\n", result);
			return 1;
	}

	platforms = malloc(num_platforms * sizeof(cl_platform_id));
	if (!platforms) {
		log_err("Platform memory allocation failed.\n");
		return 1;
	}

	/* Assume num_platforms doesn't change on the fly. */
	result = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (result != CL_SUCCESS) {
		log_err("No platforms found: %d\n", result);
		free(platforms);
		return 1;
	}
	if (o->platform >= num_platforms) {
		log_err("Specified platform %u not found\n", o->platform);
		free(platforms);
		return 1;
	}

	platform_id = platforms[o->platform];
	free(platforms);
	if (print_platform_info(platform_id) != CL_SUCCESS) {
		return 1;
	}

	result = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	if (result != CL_SUCCESS || !num_devices) {
		log_err("No device found: %d\n", result);
		return 1;
	}
	devices = malloc(num_devices * sizeof(cl_device_id));
	result = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices,  devices, NULL);
	if (result != CL_SUCCESS) {
		log_err("No device found: %d\n", result);
		return 1;
	}
	if (o->device >= num_devices) {
		log_err("Specified device %u not found\n", o->device);
		free(devices);
		return 1;
	}
	device_id = devices[o->device];
	free(devices);
	if (print_device_info(device_id)) {
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
			log_err("CreateCommandQueueWithProperties failed: %d\n", result);
			return 1;
		}
	}

	switch(io_u->ddir) {
	case DDIR_READ:
		result = clEnqueueReadBuffer(o->command_queue, o->buffer, CL_TRUE, 0,
					     io_u->buflen, buffer, 0, NULL, NULL);
		if (result != CL_SUCCESS) {
			log_err("EnqueueRead failed: %d\n", result);
			return 1;
		}
		break;
	case DDIR_WRITE:
		result = clEnqueueWriteBuffer(o->command_queue, o->buffer, CL_TRUE, 0,
					      io_u->buflen, buffer, 0, NULL, NULL);
		if (result != CL_SUCCESS) {
			log_err("EnqueueWrite failed: %d\n", result);
			return 1;
		}
		break;
	default:
		log_err("Unimplemented option\n");
		break;
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
			log_err("Host buffer allocation failed\n");
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
			log_err("CreateContext failed: %d\n", result);
			return 1;
		}
	}

	o->buffer = clCreateBuffer(o->context, flags, total_mem, o->host_buffer, &result);
	if (result != CL_SUCCESS) {
		log_err("CreateBuffer failed: %d\n", result);
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
		log_err("ReleaseMemObject failed: %d\n", result);
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
