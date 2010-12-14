#include <GL/glut.h>
#include "beam_kernel.cu"
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
extern "C"
{
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
	
	uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}
	
	void setParameters(SimParams *hostParams)
	{		
		cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
	}

	void calcHash(uint* Hash, uint* Index, float* pos, int numParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		calcHashD<<< numBlocks, numThreads >>>(Hash, Index, (float4 *) pos, numParticles);
	    
		cutilCheckMsg("Kernel execution failed");
	}

	void reorderDataAndFindCellStart(
									uint* cellStart, 
									uint* cellEnd, 
									float* sortedPos, 
									float* sortedReferencePos, 
									uint* Hash, 
									uint* Index, 
									float* oldPos, 
									float* oldReferencePos, 
									uint numParticles, 
									uint numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		cutilSafeCall(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

		#if USE_TEX
			cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, oldReferencePos, numParticles*sizeof(float4)));
		#endif

			uint smemSize = sizeof(uint)*(numThreads+1);
			reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
				cellStart,
				cellEnd,
				(float4 *) sortedPos,
				(float4 *) sortedReferencePos,
				Hash,
				Index,
				(float4 *) oldPos,
				(float4 *) oldReferencePos,
				numParticles);
			cutilCheckMsg("Kernel execution failed: reorderDataAndFindCellStartD");

		#if USE_TEX
			cutilSafeCall(cudaUnbindTexture(oldPosTex));
			cutilSafeCall(cudaUnbindTexture(oldReferencePosTex));
		#endif
	}

	void calcDensity(			
			float* measures,
			float* sortedReferencePos,						
			uint* cellStart,
			uint* cellEnd,
			uint numParticles,
			uint numGridCells)
	{

		#if USE_TEX
		cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, sortedReferencePos, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
		cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
		#endif

		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		calcDensityD<<< numBlocks, numThreads >>>(											  
											  (float4*)measures,
											  (float4*)sortedReferencePos,                                          											  
											  cellStart,
											  cellEnd,
											  numParticles);

		cutilCheckMsg("Kernel execution failed");

		#if USE_TEX
		cutilSafeCall(cudaUnbindTexture(oldReferencePosTex));
		cutilSafeCall(cudaUnbindTexture(cellStartTex));
		cutilSafeCall(cudaUnbindTexture(cellEndTex));
		#endif
	}

	void calcDensityDenominator(			
			float* measures,
			float* sortedReferencePos,						
			uint* cellStart,
			uint* cellEnd,
			uint numParticles,
			uint numGridCells)
	{

		#if USE_TEX
		cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, sortedReferencePos, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
		cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
		#endif

		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		calcDensityDenominatorD<<< numBlocks, numThreads >>>(											  
											  (float4*)measures,
											  (float4*)sortedReferencePos,                                          											  
											  cellStart,
											  cellEnd,
											  numParticles);

		cutilCheckMsg("Kernel execution failed");

		#if USE_TEX
		cutilSafeCall(cudaUnbindTexture(oldReferencePosTex));
		cutilSafeCall(cudaUnbindTexture(cellStartTex));
		cutilSafeCall(cudaUnbindTexture(cellEndTex));
		#endif
	}

	void calcDisplacementGradient(
				float* udisplacementGradient, 
				float* vdisplacementGradient, 
				float* wdisplacementGradient, 
				float* sortedPos,	
				float* sortedReferencePos,		
				float* sortedMeasures,
				uint* Index,
				uint* cellStart,
				uint* cellEnd,
				uint numParticles,
				uint numGridCells)
	{
		#if USE_TEX
		cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, sortedReferencePos, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
		cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
		#endif

		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		calcDisplacementGradientD<<< numBlocks, numThreads >>>(
			(float4*)udisplacementGradient, 
			(float4*)vdisplacementGradient, 
			(float4*)wdisplacementGradient, 
			(float4*)sortedPos,	
			(float4*)sortedReferencePos,	
			(float4*)sortedMeasures,
			Index, 
			cellStart,
			cellEnd,
			numParticles);

		cutilCheckMsg("Kernel execution failed");

		#if USE_TEX
		cutilSafeCall(cudaUnbindTexture(oldPosTex));
		cutilSafeCall(cudaUnbindTexture(oldReferencePosTex));
		cutilSafeCall(cudaUnbindTexture(oldMeasuresTex));
		cutilSafeCall(cudaUnbindTexture(cellStartTex));
		cutilSafeCall(cudaUnbindTexture(cellEndTex));
		#endif
	}

	void calcAcceleration(
				float* acceleration,
				float* sortedPos,	
				float* sortedReferencePos,	
				float* uDisplacementGradient,		
				float* vDisplacementGradient,		
				float* wDisplacementGradient,		
				float* sortedMeasures,
				uint* Index,
				uint* cellStart,
				uint* cellEnd,
				uint numParticles,
				uint numGridCells)
	{
		#if USE_TEX
		cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, sortedReferencePos, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, olduDisplacementGradientTex, uDisplacementGradient, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldvDisplacementGradientTex, vDisplacementGradient, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldwDisplacementGradientTex, wDisplacementGradient, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
		cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
		#endif

		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		calcAccelerationD<<< numBlocks, numThreads >>>(
			(float4*)acceleration, 
			(float4*)sortedPos,	
			(float4*)sortedReferencePos,	
			(float4*)uDisplacementGradient,	
			(float4*)vDisplacementGradient,	
			(float4*)wDisplacementGradient,	
			(float4*)sortedMeasures,
			Index, 
			cellStart,
			cellEnd,
			numParticles);

		cutilCheckMsg("Kernel execution failed");

		#if USE_TEX				
		cutilSafeCall(cudaUnbindTexture(oldPosTex));
		cutilSafeCall(cudaUnbindTexture(oldReferencePosTex));
		cutilSafeCall(cudaUnbindTexture(olduDisplacementGradientTex));
		cutilSafeCall(cudaUnbindTexture(oldvDisplacementGradientTex));
		cutilSafeCall(cudaUnbindTexture(oldwDisplacementGradientTex));
		cutilSafeCall(cudaUnbindTexture(oldMeasuresTex));
		cutilSafeCall(cudaUnbindTexture(cellStartTex));
		cutilSafeCall(cudaUnbindTexture(cellEndTex));
		#endif
	}

	void integrateSystem(
		float* position, 
		float* velocity, 		
		float* acceleration, 
		uint numParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);
		
		integrate<<< numBlocks, numThreads >>>(
			(float4*)position,
			(float4*)velocity,			
			(float4*)acceleration,
			numParticles);
	    		
		cutilCheckMsg("integrate kernel execution failed");
	}
}
