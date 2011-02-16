#ifndef __POISEUILLE_FLOW_SYSTEM_CUH__
#define __POISEUILLE_FLOW_SYSTEM_CUH__
#include "poiseuilleFlowKernel.cuh"
extern "C"
{		
	void setParameters(PoiseuilleParams *hostParams);	

	void integratePoiseuilleSystem(
		float* pos,
		float* vel,  
		float* velLeapFrog,
		float* acc,
		uint numParticles);

	void calculatePoiseuilleHash(
		uint*  gridParticleHash,
		uint*  gridParticleIndex,
		float* pos, 
		int    numParticles);

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
		uint   numCells);

	void calculatePoiseuilleDensity(			
		float* measures,
		float* sortedPos,
		float* sortedVel,
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells);

	void calculatePoiseuilleAcceleration(	
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
