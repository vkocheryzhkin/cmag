#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include "fluid_kernel.cu"
#include "magUtil.cuh"
extern "C"
{	
	void setParameters(SimParams *hostParams){
		cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
	}	

	void removeRightBoundary(
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

	void calculateDensityVariation(			
		float* sortedVariations,
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

			calculateDensityVariationD<<< numBlocks, numThreads >>>(										  
				(float4*)sortedVariations,
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

	void calculateDensity(			
		float* sortedMeasures,		
		float* sortedVariations,	
		uint numParticles,
		uint numGridCells){
			#if USE_TEX
			cutilSafeCall(cudaBindTexture(0, oldVariationsTex, sortedVariations, numParticles*sizeof(float4)));
			#endif
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calculateDensityD<<< numBlocks, numThreads >>>(										  
				(float4*)sortedMeasures,
				(float4*)sortedVariations,	
				numParticles);

			cutilCheckMsg("Kernel execution failed");

			#if USE_TEX
			cutilSafeCall(cudaUnbindTexture(oldVariationsTex));
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

