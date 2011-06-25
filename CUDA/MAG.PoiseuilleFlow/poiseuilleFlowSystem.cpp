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
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

using namespace thrust;

PoiseuilleFlowSystem::PoiseuilleFlowSystem(
	float deltaTime,
	uint3 fluid_size,
	float amplitude,	
	float wave_speed,
	float soundspeed,
	float3 gravity,
	int boundaryOffset,
	uint3 gridSize,
	float radius,
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
		numParticles = fluid_size.x * fluid_size.y * fluid_size.z +			
			2 * gridSize.x * boundaryOffset;
		numGridCells = gridSize.x * gridSize.y * gridSize.z;
		gridSortBits = 18;	
		cfg.fluid_size = fluid_size;
		cfg.gridSize = gridSize;	
		cfg.boundaryOffset = boundaryOffset;
		cfg.amplitude = amplitude;
		cfg.wave_speed = wave_speed;					
		cfg.radius = radius;				
		cfg.smoothingRadius = 3.0f * cfg.radius;	
		cfg.restDensity = 1000.0f;						
		cfg.particleMass = 0.0f;
		cfg.cellcount = 3;			    			
		cfg.worldOrigin = make_float3(-getHalfWorldXSize(), -getHalfWorldYSize(), -getHalfWorldZSize());
		float cellSize = cfg.radius * 2.0f;  
		cfg.cellSize = make_float3(cellSize, cellSize, cellSize);	    		

		cfg.worldSize = make_float3(
			cfg.gridSize.x * 2.0f * cfg.radius,
			cfg.gridSize.y * 2.0f * cfg.radius,
			cfg.gridSize.z * 2.0f * cfg.radius);	    

		cfg.boundaryDamping = -1.0f;
		cfg.gravity = gravity;
		cfg.soundspeed = soundspeed;
		cfg.mu = powf(10.0f, -3.0f);	
		cfg.deltaTime = deltaTime;

	/*	params.gamma = 7.0f;
		params.B = 200 * params.restDensity * abs(params.gravity.y) *		
			(2 * params.radius * fluid_size.y ) / params.gamma;	
		params.soundspeed = sqrt(params.B * params.gamma / params.restDensity);*/

		//cfg.B = 100 * cfg.restDensity * pow(cfg.soundspeed, 2) / 7.0f;

		cfg.IsBoundaryConfiguration = true;
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

void PoiseuilleFlowSystem::_initialize(uint numParticles){
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
		
		//SetColorBuffer(numParticles);

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

	setParameters(&cfg);
	
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

void PoiseuilleFlowSystem::Reset(){
	elapsedTime = 0.0f;	
	time_shift = 0.0f;
	time_relax = 1000 * cfg.deltaTime;
	currentWaveHeight = 0.0f;
	cfg.IsBoundaryConfiguration = true;
	float jitter = cfg.radius * 0.01f;			            
	float spacing = cfg.radius * 2.0f;
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

	cfg.particleMass = cfg.restDensity / CalculateMass(hPos, cfg.gridSize);
	setParameters(&cfg);	
	if (IsOpenGL)
		Coloring();
}

float PoiseuilleFlowSystem::CalculateMass(float* positions, uint3 gridSize){
	float x = positions[(gridSize.x / 2) * 4 + 0];
	float y = positions[(gridSize.x / 2) * 4 + 1];
	float2 testPoint  = make_float2(x,y);
	float sum = 0.0f;
	for(uint i = 0; i < numParticles; i++){		
		float2 tempPoint = make_float2(positions[4 * i + 0], positions[4 * i + 1]);
		float2 r = make_float2(testPoint.x - tempPoint.x, testPoint.y - tempPoint.y);
		float dist = sqrt(r.x * r.x + r.y * r.y);
		float q = dist / cfg.smoothingRadius;	
		float coeff = 7.0f / 4 / CUDART_PI_F / powf(cfg.smoothingRadius, 2);
		if(q < 2)
			sum +=coeff *(powf(1 - 0.5f * q, 4) * (2 * q + 1));
	}
	return sum;
}

void PoiseuilleFlowSystem::initFluid( float spacing, float jitter, uint numParticles){
	srand(1973);			
	int xsize = cfg.fluid_size.x;
	int ysize = cfg.fluid_size.y;
	int zsize = cfg.fluid_size.z;
	
	for(int z = 0; z < zsize; z++) {
		for(int y = 0; y < ysize; y++) {
			for(int x = 0; x < xsize; x++) {				
				uint i = (z * ysize * xsize) + y * xsize + x;
				if (i < numParticles) {
					hPos[i*4] = (spacing * x) + cfg.radius - getHalfWorldXSize();
					hPos[i*4+1] = -1.0f * cfg.fluid_size.y * cfg.radius 
						+ (spacing * y) + cfg.radius;						
					hPos[i*4+2] = (spacing * z) + cfg.radius - getHalfWorldZSize();		
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
		cfg.fluid_size.x *
		cfg.fluid_size.y * 
		cfg.fluid_size.z;

	//float sigma = 1.0f / (cfg.fluid_size.x * 2 * cfg.radius);
	//bottom
	size[0] = cfg.fluid_size.x;
	size[1] = cfg.boundaryOffset;
	size[2] = 1;	 
	for(uint z=0; z < size[2]; z++) {
		for(uint y=0; y < size[1]; y++) {
			for(uint x=0; x < size[0]; x++) {
				uint i = numAllocatedParticles + (z * size[1] * size[0]) + (y * size[0]) + x;								
				float j = cfg.radius * (2 * x + 1);
				hPos[i * 4] = j + cfg.worldOrigin.x;					 					
				hPos[i * 4 + 1] = -1.0f * cfg.fluid_size.y * cfg.radius
					- (spacing * y) - cfg.radius;
				hPos[i*4+2] = (spacing * z) + cfg.radius + cfg.worldOrigin.z;					
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
				float j = cfg.radius * (2 * x + 1);
				hPos[i * 4] = j + cfg.worldOrigin.x;			
				hPos[i*4+1] = cfg.fluid_size.y * cfg.radius
					+ (spacing * y) + cfg.radius;	
				hPos[i*4+2] = (spacing * z) + cfg.radius + cfg.worldOrigin.z;					
				hPos[i*4+3] = -2.0f * (1.0f + y);//boundary				
			}
		}
	}
}


void PoiseuilleFlowSystem::Update(){
	assert(IsInitialized);

	float *dPos;

	if (IsOpenGL) 
		dPos = (float *) mapGLBufferObject(&cuda_posvbo_resource);
	else 
		dPos = (float *) cudaPosVBO;   

	if(cfg.IsBoundaryConfiguration){
		time_shift +=cfg.deltaTime;
		if (currentWaveHeight < cfg.amplitude){
			ExtConfigureBoundary(dPos, currentWaveHeight, numParticles);
			currentWaveHeight += cfg.deltaTime * powf(10.0f, -3.0f);				
		}
		else{
			time_relax -= cfg.deltaTime;
			if(time_relax < 0){				
				cfg.IsBoundaryConfiguration = !cfg.IsBoundaryConfiguration;
				setParameters(&cfg);
			}
		}
	}			

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
		cfg.IsBoundaryConfiguration? 0: elapsedTime - time_shift,
		numGridCells);    

	computePressureForce(
		pressureForce,//not sorted		
		dMeasures, //input
		dSortedPos, 
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		cfg.IsBoundaryConfiguration? 0: elapsedTime - time_shift,
		numGridCells);

	computeCoordinates(
		dPos,
		dVel,	
		dVelLeapFrog,
		viscousForce,
		pressureForce,
		cfg.IsBoundaryConfiguration? 0: elapsedTime - time_shift,
		numParticles);

	if (IsOpenGL) {
		unmapGLBufferObject(cuda_posvbo_resource);
	}
	elapsedTime+= cfg.deltaTime;
}

void PoiseuilleFlowSystem::Coloring()
{
	uint numParticles = getNumParticles();
	host_vector<float4> h_position(numParticles);		
	host_vector<uint> h_index(numParticles);
	device_ptr<uint> index((uint*)getCudaIndex());	
	device_ptr<float4> position((float4*)mapGLBufferObject(&cuda_posvbo_resource));			

	thrust::copy(position, position + numParticles, h_position.begin());	
	thrust::copy(index, index + numParticles, h_index.begin());	
	
	unmapGLBufferObject(cuda_posvbo_resource);

	glBindBufferARB(GL_ARRAY_BUFFER, colorVBO);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	for(int i = 0; i < numParticles; i++){
		float type = h_position[i].w;		
		if(type != 0){			
			*ptr++=0; *ptr++=0; *ptr++=0;}
		else
			if (h_position[i].x < (cfg.worldOrigin.x + 4 * cfg.radius)){
				*ptr++=255; *ptr++=0; *ptr++=0;		
				//*ptr++=0; *ptr++=255; *ptr++=255;			
			}
			else{
				*ptr++=255; *ptr++=255; *ptr++=255;			
				//*ptr++=0; *ptr++=255; *ptr++=255;		
			}					
		*ptr++ = 1.0f;
	}		
	glUnmapBufferARB(GL_ARRAY_BUFFER);
}


