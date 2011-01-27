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

void integrateSystem(
			 float* pos,
             float* vel,  
			 float* displacement,
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
			 float* sortedDisplacement,
             uint*  gridParticleHash,
             uint*  gridParticleIndex,
		     float* oldPos,
		     float* oldVel,
			 float* oldDisplacement,
		     uint   numParticles,
		     uint   numCells);

void calcDensityAndPressure(			
			float* measures,
			float* sortedPos,		
			float* sortedVel,
			uint* gridParticleIndex,
			uint* cellStart,
			uint* cellEnd,
			uint numParticles,
			uint numGridCells);

void calcDisplacementGradient(
	float* duDisplacementGradient,
	float* dvDisplacementGradient,
	float* dwDisplacementGradient,
	float* dSortedPos,
	float* dSortedDisplacement,
	uint*dIndex,
	uint*dCellStart,
	uint*dCellEnd,
	uint numParticles,
	uint numGridCells);   

void calcAndApplyAcceleration(	
	float* acceleration,	
	float* duDisplacementGradient,
	float* dvDisplacementGradient,
	float* dwDisplacementGradient,
	float* measures,
	float* sortedPos,			
	float* sortedVel,
	float* sortedDisplacement,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles,
	uint numGridCells);
}

#endif