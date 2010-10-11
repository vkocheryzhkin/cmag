#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include "beam_kernel.cuh"
#include "cudpp/cudpp.h"

// Particle system class
class ParticleSystem
{
public:
    ParticleSystem(uint numParticles, uint3 gridSize, bool IsGLEnabled);
    ~ParticleSystem(); 

	enum ParticleArray
    {
        POSITION,
        VELOCITY,		
		REFERENCE_POSITION,
    };

    void update();	
	void reset();	
	void preInit();

	void setArray(ParticleArray array, const float* data, int start, int count);

	int getNumParticles() const { return numParticles; }
	unsigned int getCurrentReadBuffer() const { return posVbo; }
    unsigned int getColorBuffer()       const { return colorVBO; }

	float getParticleRadius()			const {return params.particleRadius;} 	

	void * getCudauDisplacementGradient()        const {return (void *)duDisplacementGradient;}
	void * getCudavDisplacementGradient()        const {return (void *)dvDisplacementGradient;}
	void * getCudawDisplacementGradient()        const {return (void *)dwDisplacementGradient;}

	void * getCudaAcceleration()        const {return (void *)dAcceleration;}	
	void * getCudaHash()				const {return (void *)dHash;}
	void * getCudaIndex()				const {return (void *)dIndex;}
	void * getCudaCellStart()        const {return (void *)dCellStart;}
	void * getCudaCellEnd()        const {return (void *)dCellEnd;}
protected: // methods
    ParticleSystem() {}
	uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();	

	void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: // data
    bool IsInitialized; 
	bool IsGLEnabled;
	uint numParticles;

    // CPU data
    float* hPos;    
	//float* hReferencePos;
	float* hVel;    
	//float* hDisplacement; //remove

    // GPU data
    float* dPos;   
	float* dSortedPos; //sorted
	float* dVelocity;
	//float* dDisplacement;		  //remove
	//float* dSortedDisplacement; //remove
	float* dReferencePos;
	float* dSortedReferencePos;

	float* duDisplacementGradient;
	float* dvDisplacementGradient;
	float* dwDisplacementGradient;
	float* dAcceleration;
	float* dMeasures; // dendity and volume

	uint*  dHash; 
	uint*  dIndex;
	uint*  dCellStart;   
	uint*  dCellEnd;     

	uint   gridSortBits;

	uint   posVbo;            // vertex buffer object for particle positions
    uint   colorVBO;          // vertex buffer object for colors
	float *cudaPosVBO;        
    float *cudaColorVBO; 

	struct cudaGraphicsResource *cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

	CUDPPHandle sortHandle;

	SimParams params;
	uint3 gridSize;
	uint numGridCells;
};

#endif // __PARTICLESYSTEM_H__
