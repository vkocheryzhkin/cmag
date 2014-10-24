#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/freeglut.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include <cuda_gl_interop.h>

#include "helper_timer.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"

#include "poiseuilleFlowKernel.cu"
extern "C"
{	
	void setParameters(PoiseuilleParams *hostParams){
		checkCudaErrors( cudaMemcpyToSymbol(params, hostParams, sizeof(PoiseuilleParams)) );
	}

	uint iDivUp(uint a, uint b){
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads){
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

	void cudaGLInit(int argc, char **argv)
	{   
		gpuGLDeviceInit(argc, (const char **)argv); //todo: init
		// if( checkCmdLineFlag(argc, (const char**)argv, "device") ) {
		// 	// cutilDeviceInit(argc, argv);
		// 	gpuDeviceInit(0);
		// } else {
		// 	//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() ); //todo: hiden?
		// }
	}

	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, 
							   cudaGraphicsMapFlagsNone));
	}

	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));	
	}

	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
	{
		void *ptr;
		checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
		size_t num_bytes; 
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,  
								   *cuda_vbo_resource));
		return ptr;
	}

	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
	   checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

	void allocateArray(void **devPtr, size_t size)
	{
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void freeArray(void *devPtr)
	{
		checkCudaErrors(cudaFree(devPtr));
	}

	void copyArrayToDevice(void* device, const void* host, int offset, int size)
	{
		checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
	}

	void integratePoiseuilleSystem(
		float *pos,
		float *vel,  
		float* velLeapFrog,
		float *acc,
		uint numParticles){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			integratePoiseuilleSystemD<<< numBlocks, numThreads >>>(
				(float4*)pos,
				(float4*)vel,
				(float4*)velLeapFrog,
				(float4*)acc,
				numParticles);
		    
			//checkCudaErrors("integrate kernel execution failed");
	}

	void sortParticles(uint *dHash, uint *dIndex, uint numParticles)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(dHash),
							thrust::device_ptr<uint>(dHash + numParticles),
							thrust::device_ptr<uint>(dIndex));
	}

	void calculatePoiseuilleHash(
		uint* gridParticleHash,
		uint* gridParticleIndex,
		float* pos, 
		int numParticles){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			calculatePoiseuilleHashD<<< numBlocks, numThreads >>>(
				gridParticleHash,
				gridParticleIndex,
				(float4 *) pos,
				numParticles);
		    
			//checkCudaErrors("Kernel execution failed: calculatePoiseuilleHashD");
	}

	void reorderPoiseuilleData(
		uint*  cellStart,
		uint*  cellEnd,
		float* sortedPos,
		float* sortedVel,
		uint*  gridParticleHash,
		uint*  gridParticleIndex,
		float* oldPos,
		float* oldVel,
		uint   numParticles,
		uint   numCells){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

			#if USE_TEX
				checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
				checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
			#endif

				uint smemSize = sizeof(uint)*(numThreads+1);
				reorderPoiseuilleDataD<<< numBlocks, numThreads, smemSize>>>(
					cellStart,
					cellEnd,
					(float4 *) sortedPos,
					(float4 *) sortedVel,
					gridParticleHash,
					gridParticleIndex,
					(float4 *) oldPos,
					(float4 *) oldVel,
					numParticles);
				//checkCudaErrors("Kernel execution failed: reorderPoiseuilleDataD");

			#if USE_TEX
				checkCudaErrors(cudaUnbindTexture(oldPosTex));
				checkCudaErrors(cudaUnbindTexture(oldVelTex));
			#endif
	}

	void calculatePoiseuilleDensity(			
		float* measures,
		float* sortedPos,	
		float* sortedVel,
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells){
			#if USE_TEX
			checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
			checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
			checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
			#endif

			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calculatePoiseuilleDensityD<<< numBlocks, numThreads >>>(										  
				(float4*)measures,
				(float4*)sortedPos,  
				(float4*)sortedVel,
				gridParticleIndex,
				cellStart,
				cellEnd,
				numParticles);

			//cutilCheckMsg("Kernel execution failed");
			//checkCudaErrors("Kernel execution failed");

			#if USE_TEX
			checkCudaErrors(cudaUnbindTexture(oldPosTex));
			checkCudaErrors(cudaUnbindTexture(oldVelTex));
			checkCudaErrors(cudaUnbindTexture(cellStartTex));
			checkCudaErrors(cudaUnbindTexture(cellEndTex));
			#endif
	}

	void calculatePoiseuilleAcceleration(
		float* acceleration,
		float* sortedMeasures,			
		float* sortedPos,			
		float* sortedVel,
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells){
			#if USE_TEX
			checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
			checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
			checkCudaErrors(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
			checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
			#endif

			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calculatePoiseuilleAccelerationD<<< numBlocks, numThreads >>>(
				(float4*)acceleration,
				(float4*)sortedMeasures,										  
				(float4*)sortedPos,                                          
				(float4*)sortedVel, 
				gridParticleIndex,
				cellStart,
				cellEnd,
				numParticles);

			//cutilCheckMsg("Kernel execution failed");
			//checkCudaErrors("Kernel execution failed");

			#if USE_TEX
			checkCudaErrors(cudaUnbindTexture(oldPosTex));
			checkCudaErrors(cudaUnbindTexture(oldVelTex));
			checkCudaErrors(cudaUnbindTexture(oldMeasuresTex));
			checkCudaErrors(cudaUnbindTexture(cellStartTex));
			checkCudaErrors(cudaUnbindTexture(cellEndTex));
			#endif
	}
}// extern "C"

