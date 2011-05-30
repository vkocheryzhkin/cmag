#ifndef __MAGUTIL_CUH__
#define __MAGUTIL_CUH__
typedef unsigned int uint;
#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif


extern "C"
{	
	void cudaInit(int argc, char **argv);
	void cudaGLInit(int argc, char **argv);
	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void allocateArray(void **devPtr, int size);
	void freeArray(void *devPtr);		
	void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
	void copyArrayToDevice(void* device, const void* host, int offset, int size);
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);	
}
#endif//__MAGUTIL_CUH__