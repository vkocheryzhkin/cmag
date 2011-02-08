#ifndef __BEAMSYSTEM_CUH__
#define __BEAMSYSTEM_CUH__
#include "beam_kernel.cuh"
extern "C"
{	
	void setBeamParameters(BeamParams *hostParams);

	void calculateBeamHash(uint* Hash, uint* Index, float* pos, int numParticles);

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
		uint numCells);

	void calculateBeamDensity(
		float* dMeasures, 
		float* sortedReferencePos, 
		uint* dCellStart, 
		uint* dCellEnd, 
		uint numParticles, 
		uint numGridCells);

	void calculateBeamDensityDenominator(
		float* dMeasures, 
		float* sortedReferencePos, 
		uint* dCellStart, 
		uint* dCellEnd, 
		uint numParticles, 
		uint numGridCells);	

	void calculateBeamDisplacementGradient(
		float* duDisplacementGradient,
		float* dvDisplacementGradient,
		float* dwDisplacementGradient, 
		float* sortedPos, 
		float* sortedReferencePos,
		float* sortedMeasures,
		uint* Index,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells);

	void calculateAcceleration(		
		float* acceleration,
		float* sortedPos,
		float* sortedReferencePos,
		float* duDisplacementGradient,
		float* dvDisplacementGradient,
		float* dwDisplacementGradient,
		float* sortedMeasures,
		uint* Index,
		uint* cellStart,
		uint* cellEnd,
		uint numParticles,
		uint numGridCells);

	void integrateBeamSystem(
		float* position,
		float* velocity,
		float* acceleration,
		uint numParticles);	
};
#endif //__BEAMSYSTEM_CUH__