#ifndef __BEAMSYSTEM_H__
#define __BEAMSYSTEM_H__
#include "beam_kernel.cuh"
#include "cudpp/cudpp.h"
class BeamSystem
{
public:
	BeamSystem(uint numParticles, uint3 gridSize, bool IsGLEnabled);
	~BeamSystem(); 

	enum ParticleArray{
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
	unsigned int getColorBuffer() const { return colorVBO; }

	float getParticleRadius() const {return params.particleRadius;} 	
	float getElapsedTime() const { return 0.0f; }

	void * getCudauDisplacementGradient() const {return (void *)duDisplacementGradient;}
	void * getCudavDisplacementGradient() const {return (void *)dvDisplacementGradient;}
	void * getCudawDisplacementGradient() const {return (void *)dwDisplacementGradient;}
	void * getCudaPositionVBO() const {return (void *)cudaPosVBO;}
	void * getCudaVelocity() const {return (void *)dVelocity;}
	void * getCudaSortedPosition() const {return (void *)dSortedPos;}	
	void * getCudaSortedReferencePosition() const {return (void *)dSortedReferencePos;}	
	void * getCudaMeasures() const {return (void *)dMeasures;}
	void * getCudaAcceleration() const {return (void *)dAcceleration;}	
	void * getCudaHash() const {return (void *)dHash;}
	void * getCudaIndex() const {return (void *)dIndex;}
	void * getCudaCellStart() const {return (void *)dCellStart;}
	void * getCudaCellEnd() const {return (void *)dCellEnd;}
protected: // methods
	BeamSystem() {}
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
	float* hVel;    	

	// GPU data    
	float* dSortedPos; 
	float* dVelocity;	
	float* dReferencePos;
	float* dSortedReferencePos;	 
	float* duDisplacementGradient;
	float* dvDisplacementGradient;
	float* dwDisplacementGradient;
	float* dAcceleration;		  
	float* dMeasures;//(float4) [density, denominator for normalized density, 0, normalized volume]

	uint*  dHash; 
	uint*  dIndex;
	uint*  dCellStart;   
	uint*  dCellEnd;     
	uint   gridSortBits;
	uint   posVbo;            
	uint   colorVBO;          
	float *cudaPosVBO;        
	float *cudaColorVBO; 

	struct cudaGraphicsResource *cuda_posvbo_resource; 
	struct cudaGraphicsResource *cuda_colorvbo_resource; 

	CUDPPHandle sortHandle;

	BeamParams params;
	uint3 gridSize;
	uint numGridCells;
};
#endif //__BEAMSYSTEM_H__
