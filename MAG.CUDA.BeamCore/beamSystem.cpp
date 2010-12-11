#include "beamSystem.h"
#include "beamSystem.cuh"
#include "beam_kernel.cuh"

#include <cutil_inline.h>
#include "vector_functions.h"
#include <memory.h>
#include <math.h>
#include <cstdlib>
#include <assert.h>
#include <GL/glew.h>


BeamSystem::BeamSystem(uint numParticles, uint3 gridSize, bool IsGLEnabled) :
    IsInitialized(false),   
	IsGLEnabled(IsGLEnabled),
    numParticles(numParticles),
	gridSize(gridSize),
    hPos(0),  
	hVel(0),	    
	dSortedPos(0),
	dVelocity(0),
	dReferencePos(0),
	dSortedReferencePos(0),
	duDisplacementGradient(0),
	dvDisplacementGradient(0),
	dwDisplacementGradient(0),
	dAcceleration(0),
	dMeasures(0),
	dHash(0),
	dIndex(0),
	dCellStart(0),
	dCellEnd(0)
{    	
	srand(1973);
	numGridCells = gridSize.x*gridSize.y*gridSize.z;
	gridSortBits = 18;

	params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	params.gridSize = gridSize;
	params.particleRadius = 1.0f / 64.0f;
	params.cellSize = make_float3(params.particleRadius * 2.0f, params.particleRadius * 2.0f, params.particleRadius * 2.0f);

	params.particleMass = 0.01f;
	params.smoothingRadius = 4.0f * params.particleRadius;	 	 
	params.gravity = make_float3(0.0f, -9.8f, 0.0f);    	 	

	float h = params.smoothingRadius;
	params.Poly6Kern = 315.0f / (64.0f * CUDART_PI_F * pow(h, 9.0f));
	params.SpikyKern = -45.0f /(CUDART_PI_F * pow(h, 6.0f));
	
	params.Young = 3000.0f;	
	params.Poisson = 0.49f;	
	
	params.deltaTime = 0.00005f;
    _initialize(numParticles);
}

BeamSystem::~BeamSystem()
{
    _finalize();
    numParticles = 0;
}

uint BeamSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] = {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
	    { 1.0, 1.0, 0.0, },
	    { 0.0, 1.0, 0.0, },
	    { 0.0, 1.0, 1.0, },
	    { 0.0, 0.0, 1.0, },
	    { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void BeamSystem::_initialize(int _numParticles)
{
    assert(!IsInitialized);
    numParticles = _numParticles;
    hPos = new float[numParticles*4];  
	hVel = new float[numParticles*4];  
    memset(hPos, 0, numParticles*4*sizeof(float));
	memset(hVel, 0, numParticles*4*sizeof(float));	
    
    unsigned int memSize = sizeof(float) * 4 * numParticles;   
	if(IsGLEnabled)
	{
		posVbo = createVBO(memSize);    
		registerGLBufferObject(posVbo, &cuda_posvbo_resource);

		colorVBO = createVBO(memSize);
		registerGLBufferObject(colorVBO, &cuda_colorvbo_resource);
		
		glBindBufferARB(GL_ARRAY_BUFFER, colorVBO);
		float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		float *ptr = data;
		for(uint i = 0; i < numParticles; i++) 
		{
			float t = 0.7f;
			colorRamp(t, ptr);
			ptr+=3;
			*ptr++ = 1.0f;
		}
		glUnmapBufferARB(GL_ARRAY_BUFFER);    
	}
	else 
	{ 
		cutilSafeCall( cudaMalloc( (void **)&cudaPosVBO, memSize )) ; 
		cutilSafeCall( cudaMalloc( (void **)&cudaColorVBO, memSize ));
	}
	
	allocateArray((void**)&dSortedPos, memSize);
	allocateArray((void**)&dVelocity, memSize);	
	allocateArray((void**)&dAcceleration, memSize);	
	allocateArray((void**)&dMeasures, memSize);		
	allocateArray((void**)&dReferencePos, memSize);
	allocateArray((void**)&dSortedReferencePos, memSize);			
	allocateArray((void**)&dHash, numParticles*sizeof(uint));
	allocateArray((void**)&dIndex, numParticles*sizeof(uint));
	allocateArray((void**)&dCellStart, numGridCells*sizeof(uint));
	allocateArray((void**)&dCellEnd, numGridCells*sizeof(uint));

	allocateArray((void**)&duDisplacementGradient, memSize); 
	allocateArray((void**)&dvDisplacementGradient, memSize); 
	allocateArray((void**)&dwDisplacementGradient, memSize); 
	
  		
	CUDPPConfiguration sortConfig;
	sortConfig.algorithm = CUDPP_SORT_RADIX;
	sortConfig.datatype = CUDPP_UINT;
	sortConfig.op = CUDPP_ADD;
	sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	cudppPlan(&sortHandle, sortConfig, numParticles, 1, 0);

	setParameters(&params);
    IsInitialized = true;
}

void BeamSystem::_finalize()
{
    assert(IsInitialized);

    delete [] hPos; 
	delete [] hVel; 	
	freeArray(dSortedPos);
	freeArray(dAcceleration);
	freeArray(dMeasures);	
	freeArray(dVelocity);		
	freeArray(dReferencePos);
	freeArray(dSortedReferencePos);	
	freeArray(dHash);
	freeArray(dIndex);
	freeArray(dCellStart);
	freeArray(dCellEnd);
	
	freeArray(duDisplacementGradient);
	freeArray(dvDisplacementGradient);
	freeArray(dwDisplacementGradient);
  
	if(IsGLEnabled)
	{
		unregisterGLBufferObject(cuda_posvbo_resource);
		unregisterGLBufferObject(cuda_colorvbo_resource);
		glDeleteBuffers(1, (const GLuint*)&posVbo);
		glDeleteBuffers(1, (const GLuint*)&colorVBO);    
	}
	else 
	{
        cutilSafeCall( cudaFree(cudaPosVBO) );
        cutilSafeCall( cudaFree(cudaColorVBO) );
    }

	cudppDestroyPlan(sortHandle);
}
void BeamSystem::setArray(ParticleArray array, const float* data, int start, int count)
{
    assert(IsInitialized);
 
    switch (array)
    {
    default:
    case POSITION:
        {         
			if(IsGLEnabled)
			{
				unregisterGLBufferObject(cuda_posvbo_resource);
				glBindBuffer(GL_ARRAY_BUFFER, posVbo);
				glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				registerGLBufferObject(posVbo, &cuda_posvbo_resource);    
			}
			else
			{
				copyArrayToDevice(cudaPosVBO, data, start*4*sizeof(float), count*4*sizeof(float));
			}
        }
        break; 
	case VELOCITY:
		copyArrayToDevice(dVelocity, data, start*4*sizeof(float), count*4*sizeof(float));
		break;	
	case REFERENCE_POSITION:
		copyArrayToDevice(dReferencePos, data, start*4*sizeof(float), count*4*sizeof(float));
		break;			
    }  	 
}

void BeamSystem::reset()
{
    float jitter = params.particleRadius*0.01f;			            		
	float spacing = params.particleRadius * 2.0f;
    uint gridSize[3];    
	gridSize[0] = 20;
	gridSize[1] = 1;		
	gridSize[2] = 1;
    initGrid(gridSize, spacing, jitter, numParticles);
        
    setArray(POSITION, hPos, 0, numParticles);   
	setArray(REFERENCE_POSITION, hPos, 0, numParticles);   		
	setArray(VELOCITY, hVel, 0, numParticles);   
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void BeamSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
	
	for(uint z=0; z<size[2]; z++) {
		for(uint y=0; y<size[1]; y++) {	
			for(uint x=0; x<size[0]; x++) {
				uint i = (z*size[1]*size[0]) + (y*size[0]) + x;
				if (i < numParticles) {
					hPos[i*4] =  1 + (spacing * x) + params.particleRadius - 1.0f ;//+ (frand() * 2.0f - 1.0f) * jitter;
					hPos[i*4+1] = - (spacing * y) - params.particleRadius ;//+ (frand() * 2.0f - 1.0f) * jitter;
					hPos[i*4+2] =1 + (spacing * z) + params.particleRadius - 1.0f;// + (frand() * 2.0f - 1.0f) * jitter;					
					hPos[i*4+3] = i;				
					
					hVel[i*4+0] = 0;	
					hVel[i*4+1] = 0;					
					hVel[i*4+2] = 0;															
					hVel[i*4+3] = (x == 0 && y ==0 ) ? 0 : 1;
				}
			}
		}
	}			
}


void BeamSystem::update()
{
    assert(IsInitialized);

    float *dPos;    
	if(IsGLEnabled)
		dPos = (float *) mapGLBufferObject(&cuda_posvbo_resource);
	else
		dPos = (float *) cudaPosVBO;

	calcHash(dHash, dIndex, dReferencePos, numParticles);

	cudppSort(sortHandle, dHash, dIndex, gridSortBits, numParticles);

	reorderDataAndFindCellStart(dCellStart, dCellEnd, dSortedPos, dSortedReferencePos, dHash, dIndex, dPos, dReferencePos, numParticles, numGridCells);

	calcDensity(dMeasures, dSortedReferencePos, dCellStart, dCellEnd, numParticles, numGridCells);	

	calcDisplacementGradient(
		duDisplacementGradient,
		dvDisplacementGradient,
		dwDisplacementGradient,
		dSortedPos,
		dSortedReferencePos,
		dMeasures,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);    
	
	calcAcceleration(
		dAcceleration,
		dSortedPos,
		dSortedReferencePos,
		duDisplacementGradient,
		dvDisplacementGradient,
		dwDisplacementGradient,
		dMeasures,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);    
      
	integrateSystem(
		dPos,
		dVelocity,		
		dAcceleration,
		numParticles);	    

	if(IsGLEnabled)
    unmapGLBufferObject(cuda_posvbo_resource);    
}