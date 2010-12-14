#include "fluidSystem.h"
#include "fluidSystem.cuh"
#include "fluid_kernel.cuh"

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

FluidSystem::FluidSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
	hMeasures(0),	
    m_dPos(0),
    m_dVel(0),
	dMeasures(0),	
    m_gridSize(gridSize),
    m_timer(0),
    m_solverIterations(1)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;
    
	m_params.particleRadius = 1.0f / 64.0f;		
	m_params.restDensity = 600.0f;
	m_params.particleMass = 0.02f;
	m_params.gasConstant =3.0f;
	m_params.viscosity = 3.5f;
	m_params.deltaTime = 0.005f;
	m_params.smoothingRadius = 3.0f * m_params.particleRadius;
	m_params.accelerationLimit = 200;//remove
    	
	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    float cellSize = m_params.particleRadius * 2.0f;  
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
    
    m_params.boundaryDamping = -1.0f;

    m_params.gravity = make_float3(0.0f, -6.8f, 0.0f);    	  
	m_params.Poly6Kern = 315.0f / (64.0f * CUDART_PI_F * pow(m_params.smoothingRadius, 9.0f));
	m_params.SpikyKern = (-0.5f) * -45.0f /(CUDART_PI_F * pow(m_params.smoothingRadius, 6.0f));
	m_params.LapKern = m_params.viscosity * 45.0f / (CUDART_PI_F * pow(m_params.smoothingRadius, 6.0f));	
    _initialize(numParticles);
}

FluidSystem::~FluidSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint FluidSystem::createVBO(uint size)
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

void
FluidSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
	hVelLeapFrog = new float[m_numParticles*4];		
	hMeasures = new float[m_numParticles*4];
	hAcceleration = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
	memset(hVelLeapFrog, 0, m_numParticles*4*sizeof(float));
	memset(hAcceleration, 0, m_numParticles*4*sizeof(float));	
	memset(hMeasures, 0, m_numParticles*4*sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));
    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL) {
        m_posVbo = createVBO(memSize);    
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    } else {
        cutilSafeCall( cudaMalloc( (void **)&m_cudaPosVBO, memSize )) ;
    }

    allocateArray((void**)&m_dVel, memSize);
	allocateArray((void**)&dVelLeapFrog, memSize);
	allocateArray((void**)&dAcceleration, memSize);
	allocateArray((void**)&dMeasures, memSize);

    allocateArray((void**)&m_dSortedPos, memSize);
    allocateArray((void**)&m_dSortedVel, memSize);
	
    allocateArray((void**)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void**)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void**)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void**)&m_dCellEnd, m_numGridCells*sizeof(uint));

    if (m_bUseOpenGL) {
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;
        for(uint i=0; i<m_numParticles; i++) {
            float t = 0.7f;//i / (float) m_numParticles;
    #if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
    #else
            colorRamp(t, ptr);
            ptr+=3;
    #endif
            *ptr++ = 1.0f;
        }
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    } else {
        cutilSafeCall( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*numParticles*4) );
    }

    // Create the CUDPP radix sort
    CUDPPConfiguration sortConfig;
    sortConfig.algorithm = CUDPP_SORT_RADIX;
    sortConfig.datatype = CUDPP_UINT;
    sortConfig.op = CUDPP_ADD;
    sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    cudppPlan(&m_sortHandle, sortConfig, numParticles, 1, 0);

    cutilCheckError(cutCreateTimer(&m_timer));

    setParameters(&m_params);

    m_bInitialized = true;
}

void
FluidSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
	delete [] hVelLeapFrog;	
	delete [] hMeasures;
	delete [] hAcceleration;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dVel);
	freeArray(dVelLeapFrog);	
	freeArray(dMeasures);
	freeArray(dAcceleration);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    if (m_bUseOpenGL) {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
    } else {
        cutilSafeCall( cudaFree(m_cudaPosVBO) );
        cutilSafeCall( cudaFree(m_cudaColorVBO) );
    }

    cudppDestroyPlan(m_sortHandle);
}
void FluidSystem::changeGravity() 
{ 
	m_params.gravity.y *= -1.0f; 
	setParameters(&m_params);  
}

// step the simulation
void FluidSystem::update()
{
    assert(m_bInitialized);

    float *dPos;

    if (m_bUseOpenGL) {
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
    } else {
        dPos = (float *) m_cudaPosVBO;
    }

    setParameters(&m_params); 
	
    calcHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

    cudppSort(m_sortHandle, m_dGridParticleHash, m_dGridParticleIndex, m_gridSortBits, m_numParticles);

	reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
		m_dSortedPos,		
		m_dSortedVel,
        m_dGridParticleHash,
        m_dGridParticleIndex,
		dPos,		
		dVelLeapFrog,
		m_numParticles,
		m_numGridCells);
	
	calcDensityAndPressure(		
		dMeasures,
		m_dSortedPos,			
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_numGridCells);

	calcAndApplyAcceleration(
		dAcceleration,
		dMeasures,		
		m_dSortedPos,			
		m_dSortedVel,
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_numGridCells);    

	integrateSystem(
		dPos,
		m_dVel,	
		dVelLeapFrog,
		dAcceleration,
		m_numParticles);
	
    if (m_bUseOpenGL) {
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
}

float* 
FluidSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);
 
    float* hdata = 0;
    float* ddata = 0;

    unsigned int vbo = 0;

    switch (array)
    {
    default:
    case POSITION:
        hdata = m_hPos;
        ddata = m_dPos;
        vbo = m_posVbo;
        break;
    case VELOCITY:
        hdata = m_hVel;
        ddata = m_dVel;
        break;	
    }

    copyArrayFromDevice(hdata, ddata, vbo, m_numParticles*4*sizeof(float));
    return hdata;
}

void
FluidSystem::setArray(ParticleArray array, const float* data, int start, int count)
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
    case VELOCITY:
        copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
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

void
FluidSystem::reset()
{
	float jitter = m_params.particleRadius*0.01f;			            
	uint s = (int) (powf((float) m_numParticles, 1.0f / 3.0f));
	float spacing = m_params.particleRadius * 2.0f;
	uint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = s;
	initGrid(gridSize, spacing, jitter, m_numParticles);

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);	
	setArray(MEASURES, hMeasures, 0, m_numParticles);
	setArray(ACCELERATION, hAcceleration, 0, m_numParticles);
	setArray(VELOCITYLEAPFROG, hVelLeapFrog, 0, m_numParticles);
}

void FluidSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
	srand(1973);
	for(uint z=0; z<size[2]; z++) {
		for(uint y=0; y<size[1]; y++) {
			for(uint x=0; x<size[0]; x++) {
				uint i = (z*size[1]*size[0]) + (y*size[0]) + x;
				if (i < numParticles) {
					m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;					
					m_hPos[i*4+3] = 1.0f;

					m_hVel[i*4] = 0.0f;
					m_hVel[i*4+1] = 0.0f;
					m_hVel[i*4+2] = 0.0f;
					m_hVel[i*4+3] = 0.0f;
				}
			}
		}
	}
}
