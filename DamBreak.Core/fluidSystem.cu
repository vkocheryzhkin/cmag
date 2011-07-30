#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "fluid_kernel.cu"
extern "C"
{		
	void setParameters(SimParams *hostParams){
		cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
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

	void copyArrayToDevice(void* device, const void* host, int offset, int size)
	{
		cutilSafeCall(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
	}

	void ExtChangeRightBoundary(
		float * position,
		uint numParticles){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			shiftRightBoundaryD<<< numBlocks, numThreads >>>(
				(float4*)position,
				numParticles);
		    
			cutilCheckMsg("removeRightBoundary kernel execution failed");
	}

	void sortParticles(uint *dHash, uint *dIndex, uint numParticles)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(dHash),
							thrust::device_ptr<uint>(dHash + numParticles),
							thrust::device_ptr<uint>(dIndex));
	}

	void ExtRemoveRightBoundary(
		float * position,
		uint numParticles){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			removeRightBoundaryD<<< numBlocks, numThreads >>>(
				(float4*)position,
				numParticles);
		    
			cutilCheckMsg("removeRightBoundary kernel execution failed");
	}

	void integrateSystem(
		float *pos,
		float *vel,  
		float* velLeapFrog,
		float *acc,
		uint numParticles){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			integrate<<< numBlocks, numThreads >>>(
				(float4*)pos,
				(float4*)vel,
				(float4*)velLeapFrog,
				(float4*)acc,
				numParticles);
		    
			cutilCheckMsg("integrate kernel execution failed");
	}

	void calcHash(
		uint* gridParticleHash,
		uint* gridParticleIndex,
		float* pos, 
		int numParticles){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			calcHashD<<< numBlocks, numThreads >>>(
				gridParticleHash,
				gridParticleIndex,
				(float4 *) pos,
				numParticles);
		    
			cutilCheckMsg("Kernel execution failed");
	}

	void reorderDataAndFindCellStart(
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

			cutilSafeCall(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

			#if USE_TEX
				cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
				cutilSafeCall(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
			#endif

				uint smemSize = sizeof(uint)*(numThreads+1);
				reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
					cellStart,
					cellEnd,
					(float4 *) sortedPos,
					(float4 *) sortedVel,
					gridParticleHash,
					gridParticleIndex,
					(float4 *) oldPos,
					(float4 *) oldVel,
					numParticles);
				cutilCheckMsg("Kernel execution failed: reorderDataAndFindCellStartD");

			#if USE_TEX
				cutilSafeCall(cudaUnbindTexture(oldPosTex));
				cutilSafeCall(cudaUnbindTexture(oldVelTex));
			#endif
	}	

	void calculateDamBreakDensity(			
		float* sortedMeasuresOutput,
		float* sortedMeasures,
		float* sortedPos,			
		float* sortedVel,		
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells){
			#if USE_TEX
			cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
			#endif

			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calculateDamBreakDensityD<<< numBlocks, numThreads >>>(										  
				(float4*)sortedMeasuresOutput,
				(float4*)sortedMeasures,
				(float4*)sortedPos,                                          
				(float4*)sortedVel, 
				gridParticleIndex,
				cellStart,
				cellEnd,
				numParticles);

			cutilCheckMsg("Kernel execution failed");

			#if USE_TEX
			cutilSafeCall(cudaUnbindTexture(oldPosTex));
			cutilSafeCall(cudaUnbindTexture(oldMeasuresTex));
			cutilSafeCall(cudaUnbindTexture(oldVelTex));
			cutilSafeCall(cudaUnbindTexture(cellStartTex));
			cutilSafeCall(cudaUnbindTexture(cellEndTex));			
			#endif
	}

	void calcAndApplyAcceleration(
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
			cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
			#endif

			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calcAndApplyAccelerationD<<< numBlocks, numThreads >>>(
				(float4*)acceleration,
				(float4*)sortedMeasures,										  
				(float4*)sortedPos,                                          
				(float4*)sortedVel, 
				gridParticleIndex,
				cellStart,
				cellEnd,
				numParticles);

			cutilCheckMsg("Kernel execution failed");

			#if USE_TEX
			cutilSafeCall(cudaUnbindTexture(oldPosTex));
			cutilSafeCall(cudaUnbindTexture(oldVelTex));
			cutilSafeCall(cudaUnbindTexture(oldMeasuresTex));
			cutilSafeCall(cudaUnbindTexture(cellStartTex));
			cutilSafeCall(cudaUnbindTexture(cellEndTex));
			#endif
	}
}// extern "C"

