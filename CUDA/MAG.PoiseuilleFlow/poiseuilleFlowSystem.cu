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
#include "poiseuilleFlowKernel.cu"
#include "magUtil.cuh"
extern "C"
{	
	void setParameters(PoiseuilleParams *hostParams){
		cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(PoiseuilleParams)) );
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
		    
			cutilCheckMsg("integrate kernel execution failed");
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
		    
			cutilCheckMsg("Kernel execution failed: calculatePoiseuilleHashD");
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

			cutilSafeCall(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

			#if USE_TEX
				cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
				cutilSafeCall(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
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
				cutilCheckMsg("Kernel execution failed: reorderPoiseuilleDataD");

			#if USE_TEX
				cutilSafeCall(cudaUnbindTexture(oldPosTex));
				cutilSafeCall(cudaUnbindTexture(oldVelTex));
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
			cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
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

			cutilCheckMsg("Kernel execution failed");

			#if USE_TEX
			cutilSafeCall(cudaUnbindTexture(oldPosTex));
			cutilSafeCall(cudaUnbindTexture(oldVelTex));
			cutilSafeCall(cudaUnbindTexture(cellStartTex));
			cutilSafeCall(cudaUnbindTexture(cellEndTex));
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
			cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
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

