#ifndef __FLUIDSYSTEM_H__
#define __FLUIDSYSTEM_H__

#include "fluid_kernel.cuh"
#include "vector_functions.h"
class DamBreakSystem
{
public:
	DamBreakSystem(
		uint3 fluidParticlesSize,
		int boundaryOffset,
		uint3 gridSize,
		float particleRadius,
		bool bUseOpenGL);
	~DamBreakSystem();

	enum ParticleArray
	{
		POSITION,
		VELOCITY,		
		MEASURES,
		ACCELERATION,
		VELOCITYLEAPFROG,
	};

	void update();
	void reset();

	float* getArray(ParticleArray array);
	void   setArray(ParticleArray array, const float* data, int start, int count);

	int getNumParticles() const { return numParticles; }
	float getElapsedTime() const { return elapsedTime; }
	float getHalfWorldXSize() {return params.gridSize.x * params.particleRadius;}
	float getHalfWorldYSize() {return params.gridSize.y * params.particleRadius;}
	float getHalfWorldZSize() {return params.gridSize.z * params.particleRadius;}

	unsigned int getCurrentReadBuffer() const { return posVbo; }
	unsigned int getColorBuffer()       const { return colorVBO; }

	void * getCudaPosVBO()              const { return (void *)cudaPosVBO; }
	void * getCudaVelVBO()              const { return (void *)dVel; }
	void * getCudaColorVBO()            const { return (void *)cudaColorVBO; }
	void * getCudaHash()				const {return (void *)dHash;}
	void * getCudaIndex()				const {return (void *)dIndex;}	
	void * getCudaSortedPosition()      const { return (void *)dSortedPos; }
	void * getCudaMeasures()            const { return (void *)dMeasures; }    
	void * getCudaAcceleration()        const {return (void *)dAcceleration;}	

	void changeRightBoundary();
	void removeRightBoundary();

	float getParticleRadius() { return params.particleRadius; }
	uint3 getGridSize() { return params.gridSize; }
	float3 getWorldOrigin() { return params.worldOrigin; }
	float3 getCellSize() { return params.cellSize; }
	float3 getGravity() {return params.gravity;}
protected: // methods
	DamBreakSystem() {}
	uint createVBO(uint size);

	void _initialize(int numParticles);
	void _finalize();

	void initFluid(uint *size, float spacing, float jitter, uint numParticles);
	void initBoundaryParticles(float spacing);	

protected: // data
	bool IsInitialized, IsOpenGL;
	uint numParticles;
	uint3 fluidParticlesSize;	
	float elapsedTime;

	// CPU data
	float* hPos;              // particle positions
	float* hVel;              // particle velocities
	float* hVelLeapFrog;
	
	float* hMeasures;
	float* hAcceleration;	        

	// GPU data
	float* dPos;
	float* dVel;
	float* dVelLeapFrog;
	
	float* dVariations;
	float* dMeasures;
	float* dAcceleration;	

	float* dSortedPos;
	float* dSortedVel;

	// grid data for sorting method
	uint*  dHash; // grid hash value for each particle
	uint*  dIndex;// particle index for each particle
	uint*  dCellStart;        // index of start of each cell in sorted list
	uint*  dCellEnd;          // index of end of cell

	uint   gridSortBits;

	uint   posVbo;            // vertex buffer object for particle positions
	uint   colorVBO;          // vertex buffer object for colors
    
	float *cudaPosVBO;        // these are the CUDA deviceMem Pos
	float *cudaColorVBO;      // these are the CUDA deviceMem Color

	struct cudaGraphicsResource *cuda_posvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *cuda_colorvbo_resource; // handles OpenGL-CUDA exchange	

	// params
	SimParams params;	
	uint numGridCells;    
};
#endif //__FLUIDSYSTEM_H__
