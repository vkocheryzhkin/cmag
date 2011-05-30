#ifndef __POISEUILLE_FLOW_SYSTEM_H__
#define __POISEUILLE_FLOW_SYSTEM_H__

#include "poiseuilleFlowKernel.cuh"
#include "vector_functions.h"
#include "cudpp/cudpp.h"
class PoiseuilleFlowSystem
{
public:
	PoiseuilleFlowSystem(
		float deltaTime,
		uint3 fluidParticlesSize,
		float amplitude,
		float sigma,
		float frequency,
		float soundspeed,
		float3 gravity,
		int boundaryOffset,
		uint3 gridSize,
		float particleRadius,
		bool bUseOpenGL);
	~PoiseuilleFlowSystem();

	enum ParticleArray
	{
		POSITION,
		PREDICTEDPOSITION,
		VELOCITY,		
		MEASURES,
		VISCOUSFORCE,
		PRESSUREFORCE,
		VELOCITYLEAPFROG,
	};

	void update();
	void reset();

	void   setArray(ParticleArray array, const float* data, int start, int count);

	int getNumParticles() const { return numParticles; }
	float getElapsedTime() const { return elapsedTime; }
	float getHalfWorldXSize() {return params.gridSize.x * params.particleRadius;}
	float getHalfWorldYSize() {return params.gridSize.y * params.particleRadius;}
	float getHalfWorldZSize() {return params.gridSize.z * params.particleRadius;}

	unsigned int getCurrentReadBuffer() const { return posVbo; }
	unsigned int getColorBuffer()       const { return colorVBO; }
	void setBoundaryWave();
	//void startBoundaryMotion();

	void * getCudaPosVBO()              const { return (void *)cudaPosVBO; }
	void * getCudaVelVBO()              const { return (void *)dVel; }
	void * getCudaColorVBO()            const { return (void *)cudaColorVBO; }
	void * getCudaHash()				const {return (void *)dHash;}
	void * getCudaIndex()				const {return (void *)dIndex;}	
	void * getCudaSortedPosition()      const { return (void *)dSortedPos; }
	void * getSortedVelocity()      const { return (void *)dSortedVel; }
	void * getMeasures()            const { return (void *)dMeasures; }    
	void * getViscousForce()        const {return (void *)viscousForce;}	
	void * getPressureForce()        const {return (void *)pressureForce;}		
	void * getPredictedPos()        const {return (void *)predictedPosition;}	
	void * getLeapFrogVelocity() const {return (void*) dVelLeapFrog;}	

	float getParticleRadius() { return params.particleRadius; }
	uint3 getGridSize() { return params.gridSize; }
	float3 getWorldOrigin() { return params.worldOrigin; }
	float3 getCellSize() { return params.cellSize; }
protected: // methods
	PoiseuilleFlowSystem() {}
	uint createVBO(uint size);

	void _initialize(int numParticles);
	void _finalize();

	void initFluid(float spacing, float jitter, uint numParticles);
	float CalculateMass(float* positions, uint3 gridSize);
	void initBoundaryParticles(float spacing);

protected: // data
	bool IsInitialized, IsOpenGL;
	uint numParticles;
	float currentWaveHeight;
	bool IsSetWaveBoundary;
	float elapsedTime;
	float epsDensity;

	// CPU data
	float* hPos;              
	float* hVel;              
	float* hMeasures;

	// GPU data
	float* dPos;
	float* predictedPosition;
	float* predictedVelocity;
	float* dVel;
	float* dVelLeapFrog;
	
	float* dMeasures;
	float* viscousForce;	
	float* pressureForce;

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

	CUDPPHandle sortHandle;

	// params
	PoiseuilleParams params;
	//uint3 gridSize;
	uint numGridCells;    
};
#endif //__POISEUILLE_FLOW_SYSTEM_H__
