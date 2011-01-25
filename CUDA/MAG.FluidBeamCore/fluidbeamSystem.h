#ifndef __FLUIDBEAMSYSTEM_H__
#define __FLUIDBEAMSYSTEM_H__

#include "fluidbeam_kernel.cuh"
#include "vector_functions.h"
#include "cudpp/cudpp.h"

class FluidBeamSystem
{
public:
    FluidBeamSystem(
		uint3 fluidParticlesGrid,
		uint3 beamParticlesGrid,
		int numBoundaryParticles,
		uint3 gridSize,
		float particleRadius,
		bool bUseOpenGL);
    ~FluidBeamSystem();

    enum ParticleArray
    {
        POSITION,
		REFERENCE_POSITION,
        VELOCITY,		
		MEASURES,
		ACCELERATION,
		VELOCITYLEAPFROG,
    };

    void update();
    void reset();

    float* getArray(ParticleArray array);
    void   setArray(ParticleArray array, const float* data, int start, int count);

    int    getNumParticles() const { return numParticles; }

    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    unsigned int getColorBuffer()       const { return m_colorVBO; }

	void * getCudaPositions()      const { return (void *)m_dPos; }
    void * getCudaSortedPosition()      const { return (void *)dSortedPos; }
	void * getCudaMeasures()            const { return (void *)dMeasures; }    
	void * getCudaHash()				const {return (void *)dHash;}
	void * getCudaIndex()				const {return (void *)dIndex;}	
	void * getCudaUDisplacementGradient()        const {return (void *)duDisplacementGradient;}
	void * getCudaVDisplacementGradient()        const {return (void *)dvDisplacementGradient;}
	void * getCudaWDisplacementGradient()        const {return (void *)dwDisplacementGradient;}
	void * getCudaVelocity()        const {return (void *)dVel;}	
	void * getCudaAcceleration()        const {return (void *)dAcceleration;}	

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
	void initBoundaryParticles(float spacing);	
	void initBeamGrid(float spacing, float jitter);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint numParticles;
	uint numFluidParticles;
	uint numBeamParticles;
	int boundaryOffset;
	uint3 fluidParticlesGrid;
	uint3 beamParticlesGrid;

    // CPU data
    float* hPos;              // particle positions
    float* hVel;              // particle velocities
	float* hVelLeapFrog;
	
	float* hMeasures;
	float* hAcceleration;	

    uint*  m_hParticleHash;
    uint*  m_hCellStart;
    uint*  m_hCellEnd;

    // GPU data
    float* m_dPos;
    float* dVel;
	float* dVelLeapFrog;
	
	float* dMeasures;
	float* dAcceleration;	

    float* dSortedPos;
    float* dSortedVel;

	float* dReferencePos;
	float* dSortedReferencePos;	 
	float* duDisplacementGradient;
	float* dvDisplacementGradient;
	float* dwDisplacementGradient;

    // grid data for sorting method
    uint*  dHash; // grid hash value for each particle
    uint*  dIndex;// particle index for each particle
    uint*  dCellStart;        // index of start of each cell in sorted list
    uint*  dCellEnd;          // index of end of cell

    uint   gridSortBits;

    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors
    
    float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

    struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    CUDPPHandle sortHandle;

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint numGridCells;

    uint m_timer;
};

#endif //__FLUIDBEAMSYSTEM_H__
