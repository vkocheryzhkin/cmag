#ifndef PERISTALSIS_SYSTEM_CUH_
#define PERISTALSIS_SYSTEM_CUH_
#include "peristalsisKernel.cuh"
extern "C"
{		
	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void allocateArray(void **devPtr, int size);
	void freeArray(void *devPtr);	
	void copyArrayToDevice(void* device, const void* host, int offset, int size);
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);

	void setParameters(Peristalsiscfg *hostcfg);	

	void ExtConfigureBoundary(
		float* pos,
		float currentWaveHeight,
		uint nemParticles);
	
	void calculatePeristalsisHash(
		uint*  gridParticleHash,
		uint*  gridParticleIndex,
		float* pos, 
		int    numParticles);

	void sortParticles(
		uint *dHash,
		uint *dIndex,
		uint numParticles);

	void reorderPeristalsisData(
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
		float elapsedTime,
		uint numParticles);
}//extern "C"
#endif //PERISTALSIS_SYSTEM_CUH_
