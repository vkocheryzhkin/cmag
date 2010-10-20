#ifndef BEAM_SYSTEM_H
#define BEAM_SYSTEM_H

#include "beam_kernel.cuh"

extern "C"
{	
	void cudaInit(int argc, char **argv);
	void cudaGLInit(int argc, char **argv);
	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void allocateArray(void **devPtr, int size);
	void freeArray(void *devPtr);

	void threadSync();

	void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
	void copyArrayToDevice(void* device, const void* host, int offset, int size);
	
	void setParameters(SimParams *hostParams);
	void calcHash(uint* Hash, uint* Index, float* pos, int numParticles);
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
		uint numCells);

	void calcDensity(
		float* dMeasures, 
		float* dSortedPos, 
		uint* dCellStart, 
		uint* dCellEnd, 
		uint numParticles, 
		uint numGridCells);

	void calcDisplacementGradient(
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

	void calcAcceleration(		
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

	void integrateSystem(
		float* position,
		float* velocity,
		float* acceleration,
		uint numParticles);	
};

#endif