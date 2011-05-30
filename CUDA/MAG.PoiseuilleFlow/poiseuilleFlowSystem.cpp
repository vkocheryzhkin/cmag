#include "magUtil.cuh"
#include "poiseuilleFlowSystem.h"
#include "poiseuilleFlowSystem.cuh"
#include <cutil_inline.h>
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>
#include <stdio.h>

PoiseuilleFlowSystem::PoiseuilleFlowSystem(
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
	bool bUseOpenGL) :
	IsInitialized(false),
	IsOpenGL(bUseOpenGL),    	
	hPos(0),
	hVel(0),
	hMeasures(0),
	dPos(0),
	dVel(0),
	dMeasures(0),	
	predictedPosition(0),
	viscousForce(0),
	pressureForce(0),
	elapsedTime(0.0f){		
		numParticles = fluidParticlesSize.x * fluidParticlesSize.y * fluidParticlesSize.z +			
			2 * gridSize.x * boundaryOffset;
		numGridCells = gridSize.x * gridSize.y * gridSize.z;
		gridSortBits = 18;	
		params.fluidParticlesSize = fluidParticlesSize;
		params.gridSize = gridSize;	
		params.boundaryOffset = boundaryOffset;
		params.amplitude = amplitude;
		params.frequency = frequency;
		params.sigma = sigma;		
		params.IsBoundaryMotion = false;		
		params.particleRadius = particleRadius;				
		params.smoothingRadius = 3.0f * params.particleRadius;	
		params.restDensity = 1000.0f;				
		//params.particleMass = 1000.0f / 3381298464.827586f;									
		//params.particleMass = 0.0442170704422149f;
		params.particleMass = 0.0f;
		params.cellcount = 3;			    			
		params.worldOrigin = make_float3(-getHalfWorldXSize(), -getHalfWorldYSize(), -getHalfWorldZSize());
		float cellSize = params.particleRadius * 2.0f;  
		params.cellSize = make_float3(cellSize, cellSize, cellSize);	    		

		params.worldSize = make_float3(
			params.gridSize.x * 2.0f * params.particleRadius,
			params.gridSize.y * 2.0f * params.particleRadius,
			params.gridSize.z * 2.0f * params.particleRadius);	    

		params.boundaryDamping = -1.0f;
		params.gravity = gravity;
		params.soundspeed = soundspeed;
		params.mu = powf(10.0f, -3.0f);	
		params.deltaTime = deltaTime;	

	/*	params.gamma = 7.0f;
		params.B = 200 * params.restDensity * abs(params.gravity.y) *		
			(2 * params.particleRadius * fluidParticlesSize.y ) / params.gamma;	
		params.soundspeed = sqrt(params.B * params.gamma / params.restDensity);*/

		IsSetWaveBoundary = false;
		currentWaveHeight = 0.0f;
		epsDensity = 0.01f;
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
	unsigned int memSize = sizeof(float) * 4 * numParticles;

	hPos = new float[numParticles*4];
	hVel = new float[numParticles*4];	
	hMeasures = new float[numParticles*4];
	memset(hPos, 0, numParticles*4*sizeof(float));
	memset(hVel, 0, numParticles*4*sizeof(float));
	memset(hMeasures, 0, numParticles*4*sizeof(float));	

	if (IsOpenGL) {
		posVbo = createVBO(memSize);    
		registerGLBufferObject(posVbo, &cuda_posvbo_resource);
		colorVBO = createVBO(numParticles*4*sizeof(float));
		registerGLBufferObject(colorVBO, &cuda_colorvbo_resource);
		
		glBindBufferARB(GL_ARRAY_BUFFER, colorVBO);
		float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		float *ptr = data;
		uint fluidParticles = params.fluidParticlesSize.x * params.fluidParticlesSize.y * params.fluidParticlesSize.z;
		for(int i=0; i < numParticles; i++) {
			float t = 0.5f;  
			if(i < fluidParticles)
				t = 0.7f;  
			if(((i % params.gridSize.x) == 0) && i < fluidParticles)
				t = 0.1f;    			
			colorRamp(t, ptr);
			ptr+=3;
			*ptr++ = 1.0f;
		}
		glUnmapBufferARB(GL_ARRAY_BUFFER);
	} else {
		cutilSafeCall( cudaMalloc( (void **)&cudaPosVBO, memSize ));
		cutilSafeCall( cudaMalloc( (void **)&cudaColorVBO, sizeof(float)*numParticles*4) );
	}

	allocateArray((void**)&dVel, memSize);
	allocateArray((void**)&dVelLeapFrog, memSize);
	allocateArray((void**)&viscousForce, memSize);
	allocateArray((void**)&pressureForce, memSize);
	allocateArray((void**)&dMeasures, memSize);
	allocateArray((void**)&predictedPosition, memSize);
	allocateArray((void**)&predictedVelocity, memSize);	
	allocateArray((void**)&dSortedPos, memSize);
	allocateArray((void**)&dSortedVel, memSize);	
	allocateArray((void**)&dHash, numParticles*sizeof(uint));
	allocateArray((void**)&dIndex, numParticles*sizeof(uint));
	allocateArray((void**)&dCellStart, numGridCells*sizeof(uint));
	allocateArray((void**)&dCellEnd, numGridCells*sizeof(uint));	

	CUDPPConfiguration sortConfig;
	sortConfig.algorithm = CUDPP_SORT_RADIX;
	sortConfig.datatype = CUDPP_UINT;
	sortConfig.op = CUDPP_ADD;
	sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	cudppPlan(&sortHandle, sortConfig, numParticles, 1, 0);    

	setParameters(&params);
	IsInitialized = true;
}

void PoiseuilleFlowSystem::_finalize(){
	assert(IsInitialized);

	delete [] hPos;
	delete [] hVel;	
	delete [] hMeasures;	

	freeArray(dVel);
	freeArray(dVelLeapFrog);	
	freeArray(dMeasures);
	freeArray(viscousForce);
	freeArray(pressureForce);
	freeArray(predictedPosition);
	freeArray(predictedVelocity);	
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

	cudppDestroyPlan(sortHandle);
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
	case VISCOUSFORCE:		
		copyArrayToDevice(viscousForce, data, start*4*sizeof(float), count*4*sizeof(float));
		break;
	case PRESSUREFORCE:		
		copyArrayToDevice(pressureForce, data, start*4*sizeof(float), count*4*sizeof(float));
		break;
	case VELOCITYLEAPFROG:		
		copyArrayToDevice(dVelLeapFrog, data, start*4*sizeof(float), count*4*sizeof(float));
		break;	
	case PREDICTEDPOSITION:		
		copyArrayToDevice(predictedPosition, data, start*4*sizeof(float), count*4*sizeof(float));
		break;	
	}       
}

inline float frand(){
	return rand() / (float) RAND_MAX;
}

void PoiseuilleFlowSystem::reset(){
	elapsedTime = 0.0f;	
	currentWaveHeight = 0.0f;
	IsSetWaveBoundary = false;
	float jitter = params.particleRadius * 0.01f;			            
	float spacing = params.particleRadius * 2.0f;
	initFluid(spacing, jitter, numParticles);
	initBoundaryParticles(spacing);
	memset(hMeasures, 0, numParticles*4*sizeof(float));	
	setArray(POSITION, hPos, 0, numParticles);	
	setArray(VELOCITY, hVel, 0, numParticles);	
	setArray(MEASURES, hMeasures, 0, numParticles);
	setArray(VISCOUSFORCE, hMeasures, 0, numParticles);
	setArray(PRESSUREFORCE, hMeasures, 0, numParticles);
	setArray(VELOCITYLEAPFROG, hMeasures, 0, numParticles);
	setArray(PREDICTEDPOSITION, hMeasures, 0, numParticles);

	params.particleMass = params.restDensity / CalculateMass(hPos, params.gridSize);
	setParameters(&params);
}

float PoiseuilleFlowSystem::CalculateMass(float* positions, uint3 gridSize){
	float x = positions[(gridSize.x / 2) * 4 + 0];
	float y = positions[(gridSize.x / 2) * 4 + 1];
	float2 testPoint  = make_float2(x,y);
	float sum = 0.0f;
	for(int i = 0; i < numParticles; i++){		
		float2 tempPoint = make_float2(positions[4 * i + 0], positions[4 * i + 1]);
		float2 r = make_float2(testPoint.x - tempPoint.x, testPoint.y - tempPoint.y);
		float dist = sqrt(r.x * r.x + r.y * r.y);
		float q = dist / params.smoothingRadius;	
		float coeff = 7.0f / 4 / CUDART_PI_F / powf(params.smoothingRadius, 2);
		if(q < 2)
			sum +=coeff *(powf(1 - 0.5f * q, 4) * (2 * q + 1));
	}
	return sum;
}

void PoiseuilleFlowSystem::initFluid( float spacing, float jitter, uint numParticles){
	srand(1973);			
	int xsize = params.fluidParticlesSize.x;
	int ysize = params.fluidParticlesSize.y;
	int zsize = params.fluidParticlesSize.z;
	
	for(int z = 0; z < zsize; z++) {
		for(int y = 0; y < ysize; y++) {
			for(int x = 0; x < xsize; x++) {				
				uint i = (z * ysize * xsize) + y * xsize + x;
				if (i < numParticles) {
					hPos[i*4] = (spacing * x) + params.particleRadius - getHalfWorldXSize();
					hPos[i*4+1] = (spacing * y) + params.particleRadius - getHalfWorldYSize()
						+ params.boundaryOffset * 2 * params.particleRadius + params.amplitude;						
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

	float sigma = 1.0f / (params.fluidParticlesSize.x * 2 * params.particleRadius);
	////bottom
	size[0] = params.fluidParticlesSize.x;
	size[1] = params.boundaryOffset;
	size[2] = 1;	 
	for(uint z=0; z < size[2]; z++) {
		for(uint y=0; y < size[1]; y++) {
			for(uint x=0; x < size[0]; x++) {
				uint i = numAllocatedParticles + (z * size[1] * size[0]) + (y * size[0]) + x;								
				float j = params.particleRadius * (2 * x + 1);
				hPos[i * 4] = j + params.worldOrigin.x;					 
				/*hPos[i * 4 + 1] = params.amplitude + params.amplitude * sinf(params.sigma * j) +
					(spacing * y) + params.particleRadius + params.worldOrigin.y;		*/		
				hPos[i * 4 + 1] = params.amplitude + (spacing * y) + params.particleRadius + params.worldOrigin.y;

				hPos[i*4+2] = (spacing * z) + params.particleRadius + params.worldOrigin.z;					
				hPos[i*4+3] = 2.0f * (1.0f + y);//boundary: 2,4,6				
			}
		}
	}	
	//top
	numAllocatedParticles += size[2] * size[1] * size[0];
	for(uint z=0; z < size[2]; z++) {
		for(uint y=0; y < size[1]; y++) {
			for(uint x=0; x < size[0]; x++) {
				uint i = numAllocatedParticles + (z * size[1] * size[0]) + (y * size[0]) + x;				
				float j = params.particleRadius * (2 * x + 1);
				hPos[i * 4] = j + params.worldOrigin.x;
			/*	hPos[i*4+1] = params.amplitude - params.amplitude * sinf(params.sigma * j) 
					+ spacing * y + params.particleRadius + params.worldOrigin.y
					+ params.boundaryOffset * 2 * params.particleRadius
					+ params.fluidParticlesSize.y * 2.0f * params.particleRadius;*/

				hPos[i*4+1] = (spacing * y) + params.worldOrigin.y
					+ params.boundaryOffset * 2 * params.particleRadius
					+ params.fluidParticlesSize.y * 2.0f * params.particleRadius
					+ params.amplitude + params.particleRadius;	

				hPos[i*4+2] = (spacing * z) + params.particleRadius + params.worldOrigin.z;					
				hPos[i*4+3] = -2.0f * (1.0f + y);//boundary				
			}
		}
	}
}

void PoiseuilleFlowSystem::setBoundaryWave()
{ 
	IsSetWaveBoundary = !IsSetWaveBoundary;
	elapsedTime = 0.0f;
}


void PoiseuilleFlowSystem::update(){
	assert(IsInitialized);

	float *dPos;

	if (IsOpenGL) 
		dPos = (float *) mapGLBufferObject(&cuda_posvbo_resource);
	else 
		dPos = (float *) cudaPosVBO;   

	/*if((IsSetWaveBoundary) && (currentWaveHeight < params.amplitude)){		
		if(currentWaveHeight < params.amplitude){			
			ExtSetBoundaryWave(dPos, currentWaveHeight, numParticles);
			currentWaveHeight += params.deltaTime * params.soundspeed;
		}
		else{
			IsSetWaveBoundary = !IsSetWaveBoundary;
			elapsedTime = 0.0f;
		}
	}*/

	calculatePoiseuilleHash(dHash, dIndex, dPos, numParticles);

	cudppSort(sortHandle, dHash, dIndex, gridSortBits, numParticles);

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

	computeDensityVariation(		
		dMeasures, //output
		dMeasures, //input
		dSortedPos, //input				
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells);

	computeViscousForce(
		viscousForce,//not sorted			
		dMeasures, //input
		dSortedPos,			
		dSortedVel,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		elapsedTime,
		numGridCells);    

	computePressureForce(
		pressureForce,//not sorted		
		dMeasures, //input
		dSortedPos, 
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		elapsedTime,
		numGridCells);

	computeCoordinates(
		dPos,
		dVel,	
		dVelLeapFrog,
		viscousForce,
		pressureForce,
		numParticles);

	if (IsOpenGL) {
		unmapGLBufferObject(cuda_posvbo_resource);
	}
	elapsedTime+= params.deltaTime;
}