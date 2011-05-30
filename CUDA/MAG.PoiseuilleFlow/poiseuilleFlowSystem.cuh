#ifndef __POISEUILLE_FLOW_SYSTEM_CUH__
#define __POISEUILLE_FLOW_SYSTEM_CUH__
#include "poiseuilleFlowKernel.cuh"
extern "C"
{		
	void setParameters(PoiseuilleParams *hostParams);	

	void ExtSetBoundaryWave(
		float* pos,
		float currentWaveHeight,
		uint nemParticles);
	
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
		uint   numGridCells);

	void computeDensityVariation(			
		float* measures,
		float* measuresInput,
		float* sortedPos,		
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells);

	void computeViscousForce(	
		float* viscousForce,					
		float* measures,
		float* sortedPos,			
		float* sortedVel,
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		float elapsedTime,
		uint numGridCells);	

	void computePressureForce(	
		float* pressureForce,					
		float* measures,
		float* sortedPos,					
		uint* gridParticleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		float elapsedTime,
		uint numGridCells);	

	void predictCoordinates(
		float* predictedPosition,
		float* predictedVelocity,
		float* pos,
		float* vel,  
		float* viscousForce,
		float* pressureForce,
		uint numParticles);

	void computeCoordinates(
		float* pos,
		float* vel,  
		float* velLeapFrog,
		float* viscousForce,
		float* pressureForce,
		uint numParticles);
}//extern "C"
#endif
