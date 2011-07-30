#include "poiseuilleFlowSystem.h"
#include "poiseuilleFlowSystem.cuh"
#include "poiseuilleFlowKernel.cuh"

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
PoiseuilleFlowSystem::PoiseuilleFlowSystem(
	uint3 fluidParticlesSize,
	int boundaryOffset,
	uint3 gridSize,
	float particleRadius,
	bool bUseOpenGL) :
	IsInitialized(false),
	IsOpenGL(bUseOpenGL),    	
	hPos(0),
	hVel(0),
	hMeasures(0),	
	dPos(0),
	dVel(0),
	dMeasures(0),		
	elapsedTime(0.0f){		
		numParticles = fluidParticlesSize.x * fluidParticlesSize.y * fluidParticlesSize.z +			
			2 * gridSize.x * boundaryOffset;
		numGridCells = gridSize.x * gridSize.y * gridSize.z;
		gridSortBits = 18;	//see radix sort for details
		params.fluidParticlesSize = fluidParticlesSize;
		params.gridSize = gridSize;	
		params.boundaryOffset = boundaryOffset;
	    			
		params.particleRadius = particleRadius;				
		params.smoothingRadius = 3.0f * params.particleRadius;	
		params.restDensity = 1000.0f;
				
		//see CalculateMassTest (shortly: i need to get density to be 1000, to do so I have to choose mass correctly)
		params.particleMass = 1000.0f /3381320880.551724;
					
		params.cellcount = 3;		
	    			
		params.worldOrigin = make_float3(-getHalfWorldXSize(), -getHalfWorldYSize(), -getHalfWorldZSize());
		float cellSize = params.particleRadius * 2.0f;  
		params.cellSize = make_float3(cellSize, cellSize, cellSize);
	    
		params.boundaryDamping = -1.0f;

		params.gravity = make_float3(powf(10.0f, -4.0f), 0.0f, 0.0f);    	  

		params.soundspeed = powf(10.0f, -4.0f);			
		params.mu = powf(10.0f, -3.0f);	

		params.deltaTime = powf(10.0f, -4.0f);
		_initialize(numParticles);
}

PoiseuilleFlowSystem::~PoiseuilleFlowSystem(){
	_finalize();
	numParticles = 0;
}

uint PoiseuilleFlowSystem::createVBO(uint size){
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

inline float lerp(float a, float b, float t){
	return a + t*(b-a);
}

void colorRamp(float t, float *r){
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

void PoiseuilleFlowSystem::_initialize(int numParticles){
	assert(!IsInitialized);

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

	for(uint i = 0; i < numParticles; i++) //todo: check density approximation
		hMeasures[4*i+0] = params.restDensity;

	unsigned int memSize = sizeof(float) * 4 * numParticles;

	if (IsOpenGL) {
		posVbo = createVBO(memSize);    
	registerGLBufferObject(posVbo, &cuda_posvbo_resource);
	} else {
		cutilSafeCall( cudaMalloc( (void **)&cudaPosVBO, memSize )) ;
	}

	allocateArray((void**)&dVel, memSize);
	allocateArray((void**)&dVelLeapFrog, memSize);
	allocateArray((void**)&dAcceleration, memSize);
	allocateArray((void**)&dMeasures, memSize);

	allocateArray((void**)&dSortedPos, memSize);
	allocateArray((void**)&dSortedVel, memSize);
	
	allocateArray((void**)&dHash, numParticles*sizeof(uint));
	allocateArray((void**)&dIndex, numParticles*sizeof(uint));

	allocateArray((void**)&dCellStart, numGridCells*sizeof(uint));
	allocateArray((void**)&dCellEnd, numGridCells*sizeof(uint));

	if (IsOpenGL) {
		colorVBO = createVBO(numParticles*4*sizeof(float));
	registerGLBufferObject(colorVBO, &cuda_colorvbo_resource);

		// fill color buffer
		glBindBufferARB(GL_ARRAY_BUFFER, colorVBO);
		float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		float *ptr = data;
		uint fluidParticles = params.fluidParticlesSize.x * params.fluidParticlesSize.y * params.fluidParticlesSize.z;
		for(uint i=0; i < numParticles; i++) {
			float t = 0.7f;  
			if(i < fluidParticles)
				t = 0.5f;  
			if(((i % params.gridSize.x) == 0) && i < fluidParticles)
				t = 0.2f;    			
			colorRamp(t, ptr);
			ptr+=3;
			*ptr++ = 1.0f;
		}
		glUnmapBufferARB(GL_ARRAY_BUFFER);
	} else {
		cutilSafeCall( cudaMalloc( (void **)&cudaColorVBO, sizeof(float)*numParticles*4) );
	}	   

	setParameters(&params);

	IsInitialized = true;
}

void PoiseuilleFlowSystem::_finalize(){
	assert(IsInitialized);

	delete [] hPos;
	delete [] hVel;
	delete [] hVelLeapFrog;	
	delete [] hMeasures;
	delete [] hAcceleration;    

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

	if (IsOpenGL) {
		unregisterGLBufferObject(cuda_posvbo_resource);
		glDeleteBuffers(1, (const GLuint*)&posVbo);
		glDeleteBuffers(1, (const GLuint*)&colorVBO);
	} else {
		cutilSafeCall( cudaFree(cudaPosVBO) );
		cutilSafeCall( cudaFree(cudaColorVBO) );
	}	
}
void PoiseuilleFlowSystem::changeGravity(){ 
	params.gravity.y *= -1.0f; 
	setParameters(&params);  
}

void PoiseuilleFlowSystem::update(){
	assert(IsInitialized);

	float *dPos;

	if (IsOpenGL) 
		dPos = (float *) mapGLBufferObject(&cuda_posvbo_resource);
	else 
		dPos = (float *) cudaPosVBO;    		
	
	calculatePoiseuilleHash(dHash, dIndex, dPos, numParticles);

	sortParticles(dHash, dIndex, numParticles);

	reorderPoiseuilleData(
		dCellStart,
		dCellEnd,
		dSortedPos,		
		dSortedVel,
		dHash,
		dIndex,
		dPos,		
		dVelLeapFrog,
		numParticles,
		numGridCells);
	
	calculatePoiseuilleDensity(		
		dMeasures,
		dSortedPos,	
		dSortedVel,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);

	calculatePoiseuilleAcceleration(
		dAcceleration,
		dMeasures,		
		dSortedPos,			
		dSortedVel,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);    

	integratePoiseuilleSystem(
		dPos,
		dVel,	
		dVelLeapFrog,
		dAcceleration,
		numParticles);
	
	if (IsOpenGL) {
		unmapGLBufferObject(cuda_posvbo_resource);
	}
	elapsedTime+= params.deltaTime;
}

void PoiseuilleFlowSystem::setArray(ParticleArray array, const float* data, int start, int count){
	assert(IsInitialized);
 
	switch (array)
	{
	default:
	case POSITION:
		{
			if (IsOpenGL) {
				unregisterGLBufferObject(cuda_posvbo_resource);
				glBindBuffer(GL_ARRAY_BUFFER, posVbo);
				glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				registerGLBufferObject(posVbo, &cuda_posvbo_resource);
			}else
			{
				copyArrayToDevice(cudaPosVBO, data, start*4*sizeof(float), count*4*sizeof(float));
			}
		}
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

inline float frand(){
	return rand() / (float) RAND_MAX;
}

void PoiseuilleFlowSystem::reset(){
	elapsedTime = 0.0f;
	float jitter = params.particleRadius * 0.01f;			            
	float spacing = params.particleRadius * 2.0f;
	initFluid(spacing, jitter, numParticles);
	initBoundaryParticles(spacing);

	setArray(POSITION, hPos, 0, numParticles);
	setArray(VELOCITY, hVel, 0, numParticles);	
	setArray(MEASURES, hMeasures, 0, numParticles);
	setArray(ACCELERATION, hAcceleration, 0, numParticles);
	setArray(VELOCITYLEAPFROG, hVelLeapFrog, 0, numParticles);
}

void PoiseuilleFlowSystem::initFluid( float spacing, float jitter, uint numParticles){
	srand(1973);			
	int xsize = params.fluidParticlesSize.x;
	int ysize = params.fluidParticlesSize.y;
	int zsize = params.fluidParticlesSize.z;
	
	for(uint z = 0; z < zsize; z++) {
		for(uint y = 0; y < ysize; y++) {
			for(uint x = 0; x < xsize; x++) {				
				uint i = (z * ysize * xsize) + y * xsize + x;
				if (i < numParticles) {
					hPos[i*4] = (spacing * x) + params.particleRadius - getHalfWorldXSize();
					hPos[i*4+1] = (spacing * y) + params.particleRadius - getHalfWorldYSize()
						+ params.boundaryOffset * 2 * params.particleRadius;						
					hPos[i*4+2] = (spacing * z) + params.particleRadius - getHalfWorldZSize();		
					hPos[i*4+3] = 0.0f; //fluid					
				}
			}
		}
	}
}
void PoiseuilleFlowSystem::initBoundaryParticles(float spacing)
{	
	uint size[3];	
	int numAllocatedParticles = 
		params.fluidParticlesSize.x *
		params.fluidParticlesSize.y * 
		params.fluidParticlesSize.z;
	////bottom
	size[0] = params.gridSize.x;
	size[1] = params.boundaryOffset;
	size[2] = 1;	 
	for(uint z=0; z < size[2]; z++) {
		for(uint y=0; y < size[1]; y++) {
			for(uint x=0; x < size[0]; x++) {
				uint i = numAllocatedParticles + (z * size[1] * size[0]) + (y * size[0]) + x;				
				hPos[i*4] = (spacing * x) + params.particleRadius + params.worldOrigin.x;					 
				hPos[i*4+1] = (spacing * y) + params.particleRadius + params.worldOrigin.y;
				hPos[i*4+2] = (spacing * z) + params.particleRadius + params.worldOrigin.z;					
				hPos[i*4+3] = 1.0f;//boundary				
			}
		}
	}	
	//top
	numAllocatedParticles += size[2] * size[1] * size[0];
	for(uint z=0; z < size[2]; z++) {
		for(uint y=0; y < size[1]; y++) {
			for(uint x=0; x < size[0]; x++) {
				uint i = numAllocatedParticles + (z * size[1] * size[0]) + (y * size[0]) + x;				
				hPos[i*4] = (spacing * x) + params.particleRadius + params.worldOrigin.x;					 
				hPos[i*4+1] = (spacing * y) + params.particleRadius + params.worldOrigin.y
					+ params.boundaryOffset * 2 * params.particleRadius
					+ params.fluidParticlesSize.y * 2.0f * params.particleRadius;			
				hPos[i*4+2] = (spacing * z) + params.particleRadius + params.worldOrigin.z;					
				hPos[i*4+3] = 1.0f;//boundary				
			}
		}
	}
}

