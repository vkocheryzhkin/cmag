#ifndef __FLUIDBEAMSYSTEM_H__
#define __FLUIDBEAMSYSTEM_H__

#include "fluidbeam_kernel.cuh"
#include "vector_functions.h"
#include "cudpp/cudpp.h"

class FluidBeamSystem
{
public:
    FluidBeamSystem(uint numFluidParticles,uint numBeamParticles, uint3 gridSize, bool bUseOpenGL);
    ~FluidBeamSystem();

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

    int    getNumParticles() const { return m_numParticles; }

    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    unsigned int getColorBuffer()       const { return m_colorVBO; }

    void * getCudaPosVBO()              const { return (void *)m_cudaPosVBO; }
	void * getCudaVelVBO()              const { return (void *)m_dVel; }
    void * getCudaColorVBO()            const { return (void *)m_cudaColorVBO; }
	void * getCudaHash()				const {return (void *)m_dGridParticleHash;}
	void * getCudaIndex()				const {return (void *)m_dGridParticleIndex;}	

    void changeGravity();

    float getParticleRadius() { return m_params.particleRadius; }
    uint3 getGridSize() { return m_params.gridSize; }
    float3 getWorldOrigin() { return m_params.worldOrigin; }
    float3 getCellSize() { return m_params.cellSize; }
protected: // methods
    FluidBeamSystem() {}
    uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();

    void initFluidGrid(float spacing, float jitter);
	void initBeamGrid(float spacing, float jitter);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint m_numParticles;
	uint numFluidParticles;
	uint numBeamParticles;

    // CPU data
    float* m_hPos;              // particle positions
    float* m_hVel;              // particle velocities
	float* hVelLeapFrog;
	
	float* hMeasures;
	float* hAcceleration;	

    uint*  m_hParticleHash;
    uint*  m_hCellStart;
    uint*  m_hCellEnd;

    // GPU data
    float* m_dPos;
    float* m_dVel;
	float* dVelLeapFrog;
	
	float* dMeasures;
	float* dAcceleration;	

    float* m_dSortedPos;
    float* m_dSortedVel;

    // grid data for sorting method
    uint*  m_dGridParticleHash; // grid hash value for each particle
    uint*  m_dGridParticleIndex;// particle index for each particle
    uint*  m_dCellStart;        // index of start of each cell in sorted list
    uint*  m_dCellEnd;          // index of end of cell

    uint   m_gridSortBits;

    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors
    
    float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

    struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    CUDPPHandle m_sortHandle;

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint m_numGridCells;

    uint m_timer;
};

#endif //__FLUIDBEAMSYSTEM_H__
