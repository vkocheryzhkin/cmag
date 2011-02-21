#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "poiseuilleFlowKernel.cuh"

#if USE_TEX
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float4, 1, cudaReadModeElementType> oldMeasuresTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif
__constant__ PoiseuilleParams params;

__device__ int3 calcGridPos(float3 p){
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) * 0.5f / params.particleRadius);
	gridPos.y = floor((p.y - params.worldOrigin.y) * 0.5f / params.particleRadius);
	gridPos.z = floor((p.z - params.worldOrigin.z) * 0.5f / params.particleRadius);
	return gridPos;
}

__device__ uint calcGridHash(int3 gridPos){
	gridPos.x = gridPos.x & (params.gridSize.x-1);  
	gridPos.y = gridPos.y & (params.gridSize.y-1);
	gridPos.z = gridPos.z & (params.gridSize.z-1);        
	return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

__global__ void calculatePoiseuilleHashD(
	uint*   gridParticleHash,  // output
	uint*   gridParticleIndex, // output
	float4* pos,               // input
	uint    numParticles){
		uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;
	    
		volatile float4 p = pos[index];

		int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
		uint hash = calcGridHash(gridPos);

		gridParticleHash[index] = hash;
		gridParticleIndex[index] = index;
}

__global__ void reorderPoiseuilleDataD(
	uint*   cellStart,        // output
	uint*   cellEnd,          // output
	float4* sortedPos,        // output
	float4* sortedVel,        // output
	uint *  gridParticleHash, // input
	uint *  gridParticleIndex,// input
	float4* oldPos,           // input
	float4* oldVel,           // input
	uint    numParticles){
		extern __shared__ uint sharedHash[];    // blockSize + 1 elements
		uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
		
		uint hash;
		if (index < numParticles) {
			hash = gridParticleHash[index];

			sharedHash[threadIdx.x+1] = hash;

			if (index > 0 && threadIdx.x == 0)
			{
				sharedHash[0] = gridParticleHash[index-1];
			}
		}

		__syncthreads();
		
		if (index < numParticles) {
			if (index == 0 || hash != sharedHash[threadIdx.x])
			{
				cellStart[hash] = index;
				if (index > 0)
					cellEnd[sharedHash[threadIdx.x]] = index;
			}

			if (index == numParticles - 1)
			{
				cellEnd[hash] = index + 1;
			}

			uint sortedIndex = gridParticleIndex[index];
			float4 pos = FETCH(oldPos, sortedIndex);       
			float4 vel = FETCH(oldVel, sortedIndex);       

			sortedPos[index] = pos;
			sortedVel[index] = vel;
		}
}

__device__ float sumParticlesInDomain(
	int3    gridPos,
	uint    index,
	float4  pos,
	float4* oldPos, 
	float4  vel,
	float4* oldVel,
	float4* measures,
	uint*   cellStart,
	uint*   cellEnd){
		uint gridHash = calcGridHash(gridPos);
		uint startIndex = FETCH(cellStart, gridHash);

		float sum = 0.0f;
		if (startIndex != 0xffffffff) {        // cell is not empty
			uint endIndex = FETCH(cellEnd, gridHash);
			for(uint j=startIndex; j<endIndex; j++) {				  
					float4 pos2 = FETCH(oldPos, j);
					float4 vel2 = FETCH(oldVel, j);
					float density2 = measures[j].x;
					float temp = 0.0f;

					float worldXSize= params.gridSize.x * 2.0f * params.particleRadius;				
					float3 relPos = make_float3(pos - pos2);
					if(gridPos.x < 0)
						relPos = make_float3(pos.x - (pos2.x - worldXSize),pos.y - pos2.y,pos.z - pos2.z); 
					else
						if(gridPos.x > params.gridSize.x - 1)
							relPos = make_float3(pos.x - (pos2.x + worldXSize),pos.y - pos2.y,pos.z - pos2.z);
						
					float dist = length(relPos);
					float q = dist / params.smoothingRadius;					
					
					float coeff = 7.0f / 478.0f / CUDART_PI_F / powf(params.smoothingRadius, 2);
					if(q < 1){
						sum += coeff * (powf(3 - q, 5) - 6 * powf(2 - q, 5) + 15 * powf(1 - q, 5));	
						//sum +=1;
						continue;
					}
					if(q < 2){
						sum += coeff * (powf(3 - q, 5) - 6 * powf(2 - q, 5));
						//sum +=1;
						continue;
					} 
					if(q < 3){
						sum += coeff * powf(3 - q, 5);						
						//sum +=1;
						continue;
					}										               				
			}
		}
		return sum;
}

__global__ void calculatePoiseuilleDensityD(			
	float4* measures, //output
	float4* oldPos,	  //input 
	float4* oldVel,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles){
		uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;    

		float4 pos = FETCH(oldPos, index);
		float4 vel = FETCH(oldVel, index);
		if(pos.w == 1.0f){				
				measures[index].x = params.restDensity;	
				measures[index].y = powf(params.soundspeed, 2) * params.restDensity;
				return;
		}
		int3 gridPos = calcGridPos(make_float3(pos));

		float sum = 0.0f;		
		for(int z=-params.cellcount; z<=params.cellcount; z++) {
			for(int y=-params.cellcount; y<=params.cellcount; y++) {
				for(int x=-params.cellcount; x<=params.cellcount; x++) {
					int3 neighbourPos = gridPos + make_int3(x, y, z);
					sum += sumParticlesInDomain(
							neighbourPos,
							index,
							pos,
							oldPos,
							vel,
							oldVel,
							measures,
							cellStart,
							cellEnd);
				}
			}
		}			
		float dens = sum * params.particleMass;
		measures[index].x = dens;	
		measures[index].y = powf(params.soundspeed, 2) * dens; 			
}

__device__ float4 getVelocityDiff(
	float4 iVelocity, 
	float4 iPosition, 
	float4 jVelocity,
	float4 jPosition)
{
	float bottomBoundary = params.worldOrigin.y + params.boundaryOffset * 2.0f * params.particleRadius;	
	float topBoundary = bottomBoundary + params.fluidParticlesSize.y * 2.0f * params.particleRadius;		
	if((jPosition.w == 1.0f) && (jPosition.y > topBoundary))
	{
		float distanceA = topBoundary - iPosition.y;
		float distanceB = jPosition.y - topBoundary;
		float beta = fmin(1.5f, 1 + distanceB / distanceA);
		return beta * iVelocity;
	}
	
	if((jPosition.w == 1.0f) && (jPosition.y < bottomBoundary))
	{
		float distanceA = iPosition.y - bottomBoundary;
		float distanceB = bottomBoundary - jPosition.y;
		float beta = fmin(1.5f, 1 + distanceB / distanceA);
		return beta * iVelocity;
	}
	return iVelocity - jVelocity;
}

__device__ float3 sumNavierStokesForces(
	int3    gridPos,
	uint    index,
	float4  pos,
	float4* oldPos, 
	float4  vel,
	float4* oldVel,
	float density,
	float pressure,				   
	float4* oldMeasures,
	uint*   cellStart,
	uint*   cellEnd){
		uint gridHash = calcGridHash(gridPos);
		uint startIndex = FETCH(cellStart, gridHash);
	    
		float3 tmpForce = make_float3(0.0f);
		float texp = 0.0f;
		float pexp = 0.0f;
		if (startIndex != 0xffffffff) {               
			uint endIndex = FETCH(cellEnd, gridHash);
			for(uint j=startIndex; j<endIndex; j++) {
				if (j != index) {             
					float4 pos2 = FETCH(oldPos, j);
					float4 vel2 = FETCH(oldVel, j);				
					float4 measure = FETCH(oldMeasures, j);
					float density2 = measure.x;
					float pressure2 = measure.y;				
					float tempExpr = 0.0f;

					float worldXSize= params.gridSize.x * 2.0f * params.particleRadius;				
					float3 relPos = make_float3(pos - pos2);
					if(gridPos.x < 0)
						relPos = make_float3(pos) - make_float3(pos2.x - worldXSize, pos2.y, pos2.z); 
					else
						if(gridPos.x > params.gridSize.x - 1)
							relPos = make_float3(pos) - make_float3(pos2.x + worldXSize, pos2.y, pos2.z); 
										
					float dist = length(relPos);
					float q = dist / params.smoothingRadius;				

					float coeff = -7.0f / 478.0f / CUDART_PI_F / powf(params.smoothingRadius, 3);
					float temp = 0.0f;
					float4 Vab = getVelocityDiff(vel, pos, vel2, pos2);
					if(q < 1){
						temp = coeff * (5 * powf(3 - q, 4) - 30 * powf(2 - q, 4) + 75 * powf(1 - q, 4));	
						tmpForce += -1.0f * params.particleMass *
							(pressure / powf(density,2) + pressure2 / powf(density2,2)) * 
							normalize(relPos) * temp +
							params.particleMass * (params.mu + params.mu) * 
							make_float3(Vab) / (density * density2) * 1.0f / dist * temp;
						continue;
					}
					if(q < 2){
						temp = coeff * (5 * powf(3 - q, 4) - 30 * powf(2 - q, 4));
						tmpForce += -1.0f * params.particleMass *
							(pressure / powf(density,2) + pressure2 / powf(density2,2)) * 
							normalize(relPos) * temp +
							params.particleMass * (params.mu + params.mu) * 
							make_float3(Vab) / (density * density2) * 1.0f / dist * temp;
						continue;
					} 
					if(q < 3){
						temp = coeff * (5 * powf(3 - q, 4));
						tmpForce += -1.0f * params.particleMass *
							(pressure / powf(density,2) + pressure2 / powf(density2,2)) * 
							normalize(relPos) * temp +
							params.particleMass * (params.mu + params.mu) * 
							make_float3(Vab) / (density * density2) * 1.0f / dist * temp;
						continue;
					}	
				}
			}
		}
		return tmpForce;				
}

__global__ void calculatePoiseuilleAccelerationD(
	float4* acceleration,			
	float4* oldMeasures,
	float4* oldPos,			
	float4* oldVel,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles){
		uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;    

		float4 pos = FETCH(oldPos, index);
		float4 vel = FETCH(oldVel, index);
		float4 measure = FETCH(oldMeasures,index);
		float density = measure.x;
		float pressure = measure.y;

		int3 gridPos = calcGridPos(make_float3(pos));

		float3 force = make_float3(0.0f);		
		for(int z=-params.cellcount; z<=params.cellcount; z++) {
			for(int y=-params.cellcount; y<=params.cellcount; y++) {
				for(int x=-params.cellcount; x<=params.cellcount; x++) {
					int3 neighbourPos = gridPos + make_int3(x, y, z);
					force += sumNavierStokesForces(neighbourPos, 
						index, 
						pos, 
						oldPos,
						vel,
						oldVel,
						density,
						pressure,					
						oldMeasures,
						cellStart, 
						cellEnd);
				}
			}
		}
		uint originalIndex = gridParticleIndex[index];					
		float3 acc = force;			
		acceleration[originalIndex] = make_float4(acc, 0.0f);
}

__global__ void integratePoiseuilleSystemD(
	float4* posArray,		 // input, output
	float4* velArray,		 // input, output  
	float4* velLeapFrogArray, // output
	float4* acceleration,	 // input
	uint numParticles){
		uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;          

		volatile float4 posData = posArray[index]; 
		if(posData.w == 1.0f) return;//skip boundary particle

		volatile float4 velData = velArray[index];
		volatile float4 accData = acceleration[index];
		volatile float4 velLeapFrogData = velLeapFrogArray[index];

		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);
		float3 acc = make_float3(accData.x, accData.y, accData.z);
		
		float3 nextVel = vel + (params.gravity + acc) * params.deltaTime;						

		float3 velLeapFrog = vel + nextVel;
		velLeapFrog *= 0.5f;

		vel = nextVel;   	
		pos += vel * params.deltaTime;   

		float halfWorldXSize = params.gridSize.x * params.particleRadius;			

		if(pos.x > halfWorldXSize){
			pos.x -= 2 * halfWorldXSize;
		}
		  
		posArray[index] = make_float4(pos, posData.w);
		velArray[index] = make_float4(vel, velData.w);
		velLeapFrogArray[index] = make_float4(velLeapFrog, velLeapFrogData.w);
}
