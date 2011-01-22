#ifndef FLUIDBEAM_SYSTEM_H
#define FLUIDBEAM_SYSTEM_H

#include "fluidbeam_kernel.cuh"

extern "C"
{
void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


void setParameters(SimParams *hostParams);

void calcHash(
			  uint*  gridParticleHash,
			  uint*  gridParticleIndex,
			  float* pos, 
			  int    numParticles);

void reorderDataAndFindCellStart(
			 uint*  cellStart,
		     uint*  cellEnd,
		     float* sortedPos,
			 float* sortedReferencePos,
		     float* sortedVel,
             uint*  gridParticleHash,
             uint*  gridParticleIndex,
		     float* oldPos,
			 float* oldReferencePos,
		     float* oldVel,
		     uint   numParticles,
		     uint   numCells);

void calcDensityAndPressure(			
			float* measures,
			float* sortedPositions,			
			float* sortedVelocities,	
			uint* gridParticleIndex,
			uint* cellStart,
			uint* cellEnd,
			uint numParticles,
			uint numGridCells);

void calcDisplacementGradient(
	float* duDisplacementGradient,
	float* dvDisplacementGradient,
	float* dwDisplacementGradient, 
	float* sortedPos, 
	float* sortedReferencePos,	
	uint* Index,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles,
	uint numGridCells);

void calcAcceleration(	
	float* acceleration,
	float* sortedPos,
	float* sortedReferencePos,
	float* uDisplacementGradient,
	float* vDisplacementGradient,
	float* wDisplacementGradient, 
	float* sortedVel,
	float* measures,		
	uint* Index,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles,
	uint numGridCells);

void integrateSystem(
	 float* pos,
     float* vel,  
	 float* velLeapFrog,
	 float* acc,
     uint numParticles);
}

#endif