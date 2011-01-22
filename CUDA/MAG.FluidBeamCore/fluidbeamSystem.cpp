#include "fluidbeamSystem.h"
#include "fluidbeamSystem.cuh"
#include "fluidbeam_kernel.cuh"

#include <cutil_inline.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

FluidBeamSystem::FluidBeamSystem(uint numFluidParticles,uint numBeamParticles, uint3 gridSize, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    numParticles(numFluidParticles + numBeamParticles),
	numFluidParticles(numFluidParticles),
	numBeamParticles(numBeamParticles),
    hPos(0),
    hVel(0),
	hMeasures(0),	
    m_dPos(0),
    dVel(0),
	dReferencePos(0),
	dSortedReferencePos(0),
	duDisplacementGradient(0),
	dvDisplacementGradient(0),
	dwDisplacementGradient(0),
	dMeasures(0),	
    m_gridSize(gridSize),
    m_timer(0)
{
    numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;

    gridSortBits = 18;//see radix sort for details

    m_params.gridSize = m_gridSize;
    m_params.numCells = numGridCells;
    m_params.numBodies = numParticles;
    
	m_params.particleRadius = 1.0f / 64.0f;		
	m_params.restDensity = 1000.0f;
	m_params.particleMass = 0.02f;
	m_params.gasConstant =3.0f;
	m_params.viscosity = 3.5f;	
	m_params.smoothingRadius = 3.0f * m_params.particleRadius;	
	m_params.cellcount = 2;
	m_params.accelerationLimit = 100;
    	
	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    float cellSize = m_params.particleRadius * 2.0f;  
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
    
    m_params.boundaryDamping = -1.0f;

    //m_params.gravity = make_float3(0.0f, -6.8f, 0.0f);    	  
	m_params.gravity = make_float3(0.0f, -9.8f, 0.0f);    	  
	m_params.Poly6Kern = 315.0f / (64.0f * CUDART_PI_F * pow(m_params.smoothingRadius, 9.0f));
	m_params.SpikyKern = -45.0f /(CUDART_PI_F * pow(m_params.smoothingRadius, 6.0f));
	m_params.LapKern = m_params.viscosity * 45.0f / (CUDART_PI_F * pow(m_params.smoothingRadius, 6.0f));	

	m_params.Young = 4500000.0f;	
	m_params.Poisson = 0.49f;	
	//m_params.deltaTime = 0.005f;
	m_params.deltaTime = 0.00005f;

    _initialize(numParticles);
}

FluidBeamSystem::~FluidBeamSystem()
{
    _finalize();
    numParticles = 0;
}

uint FluidBeamSystem::createVBO(uint size)
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

void FluidBeamSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    numParticles = numParticles;

    hPos = new float[numParticles*4];
    hVel = new float[numParticles*4];
	hVelLeapFrog = new float[numParticles*4];		
	hMeasures = new float[numParticles*4];
	hAcceleration = new float[numParticles*4];
    memset(hPos, 0, numParticles*4*sizeof(float));
    memset(hVel, 0, numParticles*4*sizeof(float));
	memset(hVelLeapFrog, 0, numParticles*4*sizeof(float));
	memset(hAcceleration, 0, numParticles*4*sizeof(float));	
	memset(hMeasures, 0, numParticles*4*sizeof(float));

    m_hCellStart = new uint[numGridCells];
    memset(m_hCellStart, 0, numGridCells*sizeof(uint));
    m_hCellEnd = new uint[numGridCells];
    memset(m_hCellEnd, 0, numGridCells*sizeof(uint));

    unsigned int memSize = sizeof(float) * 4 * numParticles;

    if (m_bUseOpenGL) 
	{
        m_posVbo = createVBO(memSize);    
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    } 
	else 
        cutilSafeCall( cudaMalloc( (void **)&m_cudaPosVBO, memSize )) ;

    allocateArray((void**)&dVel, memSize);
	allocateArray((void**)&dVelLeapFrog, memSize);
	allocateArray((void**)&dAcceleration, memSize);
	allocateArray((void**)&dMeasures, memSize);
	allocateArray((void**)&dReferencePos, memSize);
	allocateArray((void**)&dSortedReferencePos, memSize);			
    allocateArray((void**)&dSortedPos, memSize);
    allocateArray((void**)&dSortedVel, memSize);
	allocateArray((void**)&duDisplacementGradient, memSize); 
	allocateArray((void**)&dvDisplacementGradient, memSize); 
	allocateArray((void**)&dwDisplacementGradient, memSize); 	
    allocateArray((void**)&dHash, numParticles*sizeof(uint));
    allocateArray((void**)&dIndex, numParticles*sizeof(uint));
    allocateArray((void**)&dCellStart, numGridCells*sizeof(uint));
    allocateArray((void**)&dCellEnd, numGridCells*sizeof(uint));

    if (m_bUseOpenGL) 
	{
        m_colorVBO = createVBO(numParticles*4*sizeof(float));
		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;
        for(uint i=0; i<numParticles; i++) 
		{
			float t =0.0f;
			if(i < numFluidParticles)
				t = 0.7f;
			else
				t = 0.3f;
            colorRamp(t, ptr);
            ptr+=3;
            *ptr++ = 1.0f;
        }
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    } 
	else 
        cutilSafeCall( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*numParticles*4) );

    // Create the CUDPP radix sort
    CUDPPConfiguration sortConfig;
    sortConfig.algorithm = CUDPP_SORT_RADIX;
    sortConfig.datatype = CUDPP_UINT;
    sortConfig.op = CUDPP_ADD;
    sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    cudppPlan(&sortHandle, sortConfig, numParticles, 1, 0);

    cutilCheckError(cutCreateTimer(&m_timer));

    setParameters(&m_params);

    m_bInitialized = true;
}

void FluidBeamSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] hPos;
    delete [] hVel;
	delete [] hVelLeapFrog;	
	delete [] hMeasures;
	delete [] hAcceleration;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(dVel);
	freeArray(dVelLeapFrog);	
	freeArray(dMeasures);
	freeArray(dAcceleration);
    freeArray(dSortedPos);
    freeArray(dSortedVel);

    freeArray(dHash);
    freeArray(dIndex);
    freeArray(dCellStart);
    freeArray(dCellEnd);

    if (m_bUseOpenGL) {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
    } else {
        cutilSafeCall( cudaFree(m_cudaPosVBO) );
        cutilSafeCall( cudaFree(m_cudaColorVBO) );
    }

    cudppDestroyPlan(sortHandle);
}
void FluidBeamSystem::changeGravity() 
{ 
	m_params.gravity.y *= -1.0f; 
	setParameters(&m_params);  
}

// step the simulation
void FluidBeamSystem::update()
{
    assert(m_bInitialized);

    float *dPos;

    if (m_bUseOpenGL) {
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
    } else {
        dPos = (float *) m_cudaPosVBO;
    }

    setParameters(&m_params); 
	
    calcHash(dHash, dIndex, dPos, numParticles);

    cudppSort(sortHandle, dHash, dIndex, gridSortBits, numParticles);

	reorderDataAndFindCellStart(
        dCellStart,
        dCellEnd,
		dSortedPos,
		dSortedReferencePos,
		dSortedVel,		
        dHash,
        dIndex,
		dPos,		
		dReferencePos,
		dVelLeapFrog,
		numParticles,
		numGridCells);
	
	calcDensityAndPressure(		
		dMeasures,
		dSortedPos,			
		dSortedVel,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);		

	calcDisplacementGradient(
		duDisplacementGradient,
		dvDisplacementGradient,
		dwDisplacementGradient,
		dSortedPos,
		dSortedReferencePos,		
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);    
	

	/*calcAcceleration(
		dAcceleration,
		dMeasures,		
		dSortedPos,			
		dSortedVel,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);   */

	calcAcceleration(
		dAcceleration,
		dSortedPos,
		dSortedReferencePos,
		duDisplacementGradient,
		dvDisplacementGradient,
		dwDisplacementGradient,
		dSortedVel,
		dMeasures,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells); 

	integrateSystem(
		dPos,
		dVel,	
		dVelLeapFrog,
		dAcceleration,
		numParticles);
	
    if (m_bUseOpenGL) {
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
}

float* FluidBeamSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);
 
    float* hdata = 0;
    float* ddata = 0;

    unsigned int vbo = 0;

    switch (array)
    {
		default:
		case POSITION:
			hdata = hPos;
			ddata = m_dPos;
			vbo = m_posVbo;
			break;
		case VELOCITY:
			hdata = hVel;
			ddata = dVel;
			break;	
    }

    copyArrayFromDevice(hdata, ddata, vbo, numParticles*4*sizeof(float));
    return hdata;
}

void FluidBeamSystem::setArray(ParticleArray array, const float* data, int start, int count)
{
    assert(m_bInitialized);
 
    switch (array)
    {
    default:
    case POSITION:
        {
            if (m_bUseOpenGL) {
                unregisterGLBufferObject(m_cuda_posvbo_resource);
                glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
            }else
			{
				copyArrayToDevice(m_cudaPosVBO, data, start*4*sizeof(float), count*4*sizeof(float));
			}
        }
        break;
	case REFERENCE_POSITION:
		copyArrayToDevice(dReferencePos, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;	
    case VELOCITY:
        copyArrayToDevice(dVel, data, start*4*sizeof(float), count*4*sizeof(float));
        break;	
	case MEASURES:
		copyArrayToDevice(dMeasures, data, start*4*sizeof(float), count*4*sizeof(float));
		break;
	case ACCELERATION:		
		copyArrayToDevice(dAcceleration, data, start*4*sizeof(float), count*4*sizeof(float));
		break;
	case VELOCITYLEAPFROG:		
		copyArrayToDevice(dVelLeapFrog, data, start*4*sizeof(float), count*4*sizeof(float));
		break;		
    }       
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void FluidBeamSystem::reset()
{
	float jitter = m_params.particleRadius*0.01f;			            	
	float spacing = m_params.particleRadius * 2.0f;	
	initFluidGrid( spacing, jitter);
	initBeamGrid(spacing, jitter);

    setArray(POSITION, hPos, 0, numParticles);
	setArray(REFERENCE_POSITION, hPos, 0, numParticles);   		
    setArray(VELOCITY, hVel, 0, numParticles);	
	setArray(MEASURES, hMeasures, 0, numParticles);
	setArray(ACCELERATION, hAcceleration, 0, numParticles);
	setArray(VELOCITYLEAPFROG, hVelLeapFrog, 0, numParticles);
}

void FluidBeamSystem::initFluidGrid(float spacing, float jitter)
{
	srand(1973);
	uint size[3];
	uint s = (int) (powf((float) numFluidParticles, 1.0f / 3.0f));
	size[0] = size[1] = size[2] = s;
	for(uint z=0; z<size[2]; z++) {
		for(uint y=0; y<size[1]; y++) {
			for(uint x=0; x<size[0]; x++) {
				uint i = (z*size[1]*size[0]) + (y*size[0]) + x;
				if (i < numFluidParticles) {
					hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;					
					hPos[i*4+3] = 1.0f;

					hVel[i*4] = 0.0f;
					hVel[i*4+1] = 0.0f;
					hVel[i*4+2] = 0.0f;
					hVel[i*4+3] = 0.0f;
				}
			}
		}
	}
}
//void FluidBeamSystem::initBeamGrid(float spacing, float jitter)
//{
//	srand(1973);	
//	//todo: exctract const
//	int zsize = 25;
//	int ysize = 32;
//	for(uint z = 0; z < zsize; z++) {
//		for(uint y = 0; y < ysize; y++) {
//				uint x = 0;
//				uint i = numFluidParticles + (z * ysize) + y;
//				if (i < numParticles) {
//					m_hPos[i*4] = (spacing * x) + m_params.particleRadius + 0.7f + (frand() * 2.0f - 1.0f) * jitter;
//					m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
//					m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;					
//					m_hPos[i*4+3] = 1.0f;
//
//					m_hVel[i*4] = 0.0f;
//					m_hVel[i*4+1] = 0.0f;
//					m_hVel[i*4+2] = 0.0f;
//					m_hVel[i*4+3] = 0.0f;
//			}
//		}
//	}
//}

void FluidBeamSystem::initBeamGrid(float spacing, float jitter)
{	
	int xsize = 20;
	int ysize = 1;
	int zsize = 5;
	for(uint z=0; z < zsize; z++) {
		for(uint y=0; y < ysize; y++) {	
			for(uint x=0; x < xsize; x++) {
				uint i = (z* ysize * xsize) + (y * xsize) + x;
				if (i < numParticles) {
					hPos[i*4] =  0.7 + (spacing * x) + m_params.particleRadius - 1.0f ;//+ (frand() * 2.0f - 1.0f) * jitter;
					hPos[i*4+1] = - (spacing * y) - m_params.particleRadius ;//+ (frand() * 2.0f - 1.0f) * jitter;
					hPos[i*4+2] =1 + (spacing * z) + m_params.particleRadius - 1.0f;// + (frand() * 2.0f - 1.0f) * jitter;					
					hPos[i*4+3] = i;				

					hVel[i*4+0] = 0;	
					hVel[i*4+1] = 0;					
					hVel[i*4+2] = 0;															
					hVel[i*4+3] = (x == 0 ) ? 0 : 1;
				}
			}
		}
	}	
}
