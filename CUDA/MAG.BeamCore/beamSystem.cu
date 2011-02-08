#include <GL/glut.h>
#include "beam_kernel.cu"
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include "magUtil.cuh"
extern "C"
{						
	void setBeamParameters(BeamParams *hostParams){		
		cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(BeamParams)) );
	}

	void calculateBeamHash(uint* Hash, uint* Index, float* pos, int numParticles){
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		calculateBeamHashD<<< numBlocks, numThreads >>>(Hash, Index, (float4 *) pos, numParticles);
	    
		cutilCheckMsg("Kernel execution failed");
	}

	void reorderBeamData(
		uint* cellStart, 
		uint* cellEnd, 
		float* sortedPos, 
		float* sortedReferencePos, 
		uint* Hash, 
		uint* Index, 
		float* oldPos, 
		float* oldReferencePos, 
		uint numParticles, 
		uint numCells){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);

			cutilSafeCall(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));		

			#if USE_TEX
				cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
				cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, oldReferencePos, numParticles*sizeof(float4)));
			#endif

				uint smemSize = sizeof(uint)*(numThreads+1);
				reorderBeamDataD<<< numBlocks, numThreads, smemSize>>>(
					cellStart,
					cellEnd,
					(float4 *) sortedPos,
					(float4 *) sortedReferencePos,
					Hash,
					Index,
					(float4 *) oldPos,
					(float4 *) oldReferencePos,
					numParticles);
				cutilCheckMsg("Kernel execution failed: reorderBeamDataD");

			#if USE_TEX
				cutilSafeCall(cudaUnbindTexture(oldPosTex));
				cutilSafeCall(cudaUnbindTexture(oldReferencePosTex));
			#endif
	}

	void calculateBeamDensity(			
		float* measures,
		float* sortedReferencePos,						
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells){
			#if USE_TEX
			cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, sortedReferencePos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
			#endif

			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calculateBeamDensityD<<< numBlocks, numThreads >>>(											  
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

	void calculateBeamDensityDenominator(			
		float* measures,
		float* sortedReferencePos,						
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells){
			#if USE_TEX
			cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, sortedReferencePos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
			#endif

			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calculateBeamDensityDenominatorD<<< numBlocks, numThreads >>>(											  
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

	void calculateBeamDisplacementGradient(
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
		uint numGridCells){
			#if USE_TEX
			cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldReferencePosTex, sortedReferencePos, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
			cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
			cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
			#endif

			uint numThreads, numBlocks;
			computeGridSize(numParticles, 64, numBlocks, numThreads);

			calculateBeamDisplacementGradientD<<< numBlocks, numThreads >>>(
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

	void calculateAcceleration(
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
		uint numGridCells){
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

			calculateAccelerationD<<< numBlocks, numThreads >>>(
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

	void integrateBeamSystem(
		float* position, 
		float* velocity, 		
		float* acceleration, 
		uint numParticles){
			uint numThreads, numBlocks;
			computeGridSize(numParticles, 256, numBlocks, numThreads);
			
			integrateBeamSystemD<<< numBlocks, numThreads >>>(
				(float4*)position,
				(float4*)velocity,			
				(float4*)acceleration,
				numParticles);
		    		
			cutilCheckMsg("integrate kernel execution failed");
	}
};
