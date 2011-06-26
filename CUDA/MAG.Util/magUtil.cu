#include <cutil_inline.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
typedef unsigned int uint;

extern "C"
{
	uint iDivUp(uint a, uint b){
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads){
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

	void cudaInit(int argc, char **argv)
	{   		
		if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
			cutilDeviceInit(argc, argv);
		} else {
			cudaSetDevice( cutGetMaxGflopsDeviceId() );
		}
	}

	void cudaGLInit(int argc, char **argv)
	{   
		if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
			cutilDeviceInit(argc, argv);
		} else {
			cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
		}
	}

	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
	{
		cutilSafeCall(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, 
							   cudaGraphicsMapFlagsNone));
	}

	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		cutilSafeCall(cudaGraphicsUnregisterResource(cuda_vbo_resource));	
	}

	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
	{
		void *ptr;
		cutilSafeCall(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
		size_t num_bytes; 
		cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,  
								   *cuda_vbo_resource));
		return ptr;
	}

	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
	   cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

	void allocateArray(void **devPtr, size_t size)
	{
		cutilSafeCall(cudaMalloc(devPtr, size));
	}

	void freeArray(void *devPtr)
	{
		cutilSafeCall(cudaFree(devPtr));
	}

	void copyArrayFromDevice(void* host, const void* device, 
			 struct cudaGraphicsResource **cuda_vbo_resource, int size)
	{   
		if (cuda_vbo_resource)
		device = mapGLBufferObject(cuda_vbo_resource);

		cutilSafeCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
	    
		if (cuda_vbo_resource)
		unmapGLBufferObject(*cuda_vbo_resource);
	}

	void copyArrayToDevice(void* device, const void* host, int offset, int size)
	{
		cutilSafeCall(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
	}

	
}