#ifndef FLUID_SYSTEM_CUH
#define FLUID_SYSTEM_CUH
#include "fluid_kernel.cuh"
extern "C"
{		
	void setParameters(SimParams *hostParams);

	void integrateSystem(
		float* pos,
		float* vel,  
		float* velLeapFrog,
		float* acc,
		uint numParticles);

	void calcHash(
		uint*  gridParticleHash,
		uint*  gridParticleIndex,
		float* pos, 
		int    numParticles);

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
		uint   numCells);

	void calculateDensityVariation(			
		float* measures,
		float* sortedPos,			
		float* sortedVel,
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells);

	void calculateDensity(			
		float* measures,		
		uint numParticles,
		uint numGridCells);

	void calcAndApplyAcceleration(	
		float* acceleration,			
		float* measures,
		float* sortedPos,			
		float* sortedVel,
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells);
}//extern "C"
#endif
