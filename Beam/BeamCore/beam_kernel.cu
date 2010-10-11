#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include "cutil_math.h"
#include "beam_kernel.cuh"

texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldReferencePosTex;
texture<float4, 1, cudaReadModeElementType> olduDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldvDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldwDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldMeasuresTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;

__constant__ SimParams params;

__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);        
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

__global__ void calcHashD(uint* Hash,  // output
               uint* Index, // output
               float4* pos, // input
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
    volatile float4 p = pos[index];

    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    Hash[index] = hash;
    Index[index] = index;
}

__global__ void reorderDataAndFindCellStartD(
								  uint*   cellStart,// output: 
							      uint*   cellEnd,             // output: 
								  float4* sortedPos,		   // output;
  							      float4* sortedReferencePos,  // output:						  
                                  uint *  Hash,				   // input: 
                                  uint *  Index,			   // input: 
								  float4* oldPos,		   // input;
								  float4* oldReferencePos,
							      uint    numParticles)
{
	extern __shared__ uint sharedHash[];    
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
    uint hash;   
    if (index < numParticles) {
        hash = Hash[index];
        
	    sharedHash[threadIdx.x+1] = hash;

	    if (index > 0 && threadIdx.x == 0)
	    {		    
		    sharedHash[0] = Hash[index-1];
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

	    uint sortedIndex = Index[index];		 				
		sortedPos[index] = FETCH(oldPos, sortedIndex);
        sortedReferencePos[index] = FETCH(oldReferencePos, sortedIndex);
	}
}


__device__ float sumDensityPart(
				   int3    gridPos,
                   uint    index,
                   float3  pos,
                   float4* oldPos, 
                   uint*   cellStart,
                   uint*   cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    uint startIndex = FETCH(cellStart, gridHash);

    float sum = 0.0f;
    if (startIndex != 0xffffffff) {                
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {             
	            float3 pos2 = make_float3(FETCH(oldPos, j));
				float wpolyExpr = 0.0f;

				float3 relPos = pos2 - pos; 
				float dist = length(relPos);

				if (dist < params.smoothingRadius) {
					wpolyExpr = pow(params.smoothingRadius,2)- pow(dist,2);					
					sum += pow(wpolyExpr,3);
				}                
            }
        }
    }
    return sum;
}

__global__ 
void calcDensityD(			
			float4* measures, //output
			float4* oldPos,	 //input sorted position					
			uint* cellStart,
			uint* cellEnd,
			uint numParticles)
			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    

	float3 pos = make_float3(FETCH(oldPos, index));
    int3 gridPos = calcGridPos(pos);

    float sum = 0.0f;
	int cellcount = 2;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                sum += sumDensityPart(neighbourPos, index, pos, oldPos, cellStart, cellEnd);
            }
        }
    }	
	float dens = sum * params.particleMass * params.Poly6Kern;
    measures[index].x = dens;	//density	
	measures[index].y = params.particleMass / dens;	//volume
}

__device__ float12 sumDisplacementGradientPart(
				   int3    gridPos,
                   uint    index,
                   float3  pos_i,
                   float4* oldPos, 
				   float3  referencePos_i,
                   float4* oldReferencePos, 
				   float4* oldMeasures,
                   uint*   cellStart,
                   uint*   cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

    uint startIndex = FETCH(cellStart, gridHash);    	
	float12 gradient = make_float12();	
	
    if (startIndex != 0xffffffff) {               
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {             
	            float3 pos_j = make_float3(FETCH(oldPos, j));				
				float3 referencePos_j = make_float3(FETCH(oldReferencePos,j));
				float volume_j = FETCH(oldMeasures, j).y;

				float3 relPos = referencePos_i - referencePos_j;
				float dist = length(relPos);

				if (dist < params.smoothingRadius) {
					float tempExpr =  (params.smoothingRadius - dist);															

					gradient.u.x += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * (relPos.x / dist) * tempExpr * tempExpr;
					gradient.u.y += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * (relPos.y / dist) * tempExpr * tempExpr;
					gradient.u.z += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * (relPos.z / dist) * tempExpr * tempExpr;
					
					gradient.v.x += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * (relPos.x / dist) * tempExpr * tempExpr;
					gradient.v.y += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * (relPos.y / dist) * tempExpr * tempExpr;
					gradient.v.z += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * (relPos.z / dist) * tempExpr * tempExpr;

					gradient.w.x += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * (relPos.x / dist) * tempExpr * tempExpr;
					gradient.w.y += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * (relPos.y / dist) * tempExpr * tempExpr;
					gradient.w.z += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * (relPos.z / dist) * tempExpr * tempExpr;					
				}                
            }
        }
    }
	return gradient;		
}

__global__ void calcDisplacementGradientD(
						  float4* udisplacementGradient,
						  float4* vdisplacementGradient,
						  float4* wdisplacementGradient,
						  float4* oldPos,	
						  float4* oldReferencePos,	
						  float4* oldMeasures,
						  uint* Index, 
						  uint* cellStart,
						  uint* cellEnd,
						  uint numParticles)			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    

	float3 pos = make_float3(FETCH(oldPos, index));
	float3 referencePos = make_float3(FETCH(oldReferencePos, index));
    int3 gridPos = calcGridPos(pos);
	float12 result = make_float12();	
	float12 buf = make_float12();	
	int cellcount = 2;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                buf = sumDisplacementGradientPart(neighbourPos, index, pos, oldPos, referencePos, oldReferencePos, oldMeasures, cellStart, cellEnd);
				result.u+=buf.u;//todo: remove this stuff
				result.v+=buf.v;
				result.w+=buf.w;
            }
        }
    }    				
	udisplacementGradient[index] = result.u;
	vdisplacementGradient[index] = result.v;
	wdisplacementGradient[index] = result.w;
}

__device__ float3 sumForcePart(
				   int3    gridPos,
                   uint    index,
                   float3  referencePos_i,
                   float4* oldReferencePos, 
				   float3  du_i,
				   float3  dv_i,
				   float3  dw_i,
                   float   volume_i, 
				   float4* oldMeasures,
                   uint*   cellStart,
                   uint*   cellEnd)
{
	uint gridHash = calcGridHash(gridPos);
    uint startIndex = FETCH(cellStart, gridHash);    
	float3 tmpForce = make_float3(0.0f);	
	float3 uSigma = make_float3(0.0f);
	float3 vSigma = make_float3(0.0f);
	float3 wSigma = make_float3(0.0f);
	float3 d = make_float3(0.0f);	
	
	float3 uJ = make_float3(du_i.x + 1, du_i.y    , du_i.z	 );
	float3 vJ = make_float3(dv_i.x	 , dv_i.y + 1, dv_i.z	 );
	float3 wJ = make_float3(dw_i.x	 , dw_i.y	 , dw_i.z + 1);

	//Green-Saint-Venant strain tensor
	float3 uE = 0.5 * make_float3(uJ.x * uJ.x + vJ.x * vJ.x + wJ.x * wJ.x - 1,	uJ.x * uJ.y + vJ.x * vJ.y + wJ.x * wJ.y	   ,	uJ.x * uJ.z + vJ.x * vJ.z + wJ.x * wJ.z		);
	float3 vE = 0.5 * make_float3(uJ.y * uJ.x + vJ.y * vJ.x + wJ.y * wJ.x    ,	uJ.y * uJ.y + vJ.y * vJ.y + wJ.y * wJ.y - 1,	uJ.y * uJ.z + vJ.y * vJ.z + wJ.y * wJ.z		);
	float3 wE = 0.5 * make_float3(uJ.z * uJ.x + vJ.z * vJ.x + wJ.z * wJ.x	 ,	uJ.z * uJ.y + uJ.z * vJ.y + wJ.z * wJ.y	   ,	uJ.z * uJ.z + vJ.z * vJ.z + wJ.z * wJ.z - 1 );

	//Stress tensor
	uSigma.x = (params.Young / ( 1 + params.Poisson))*(uE.x + (params.Poisson / ( 1 - 2*params.Poisson))*(uE.x + vE.y + wE.z));
	vSigma.y = (params.Young / ( 1 + params.Poisson))*(vE.y + (params.Poisson / ( 1 - 2*params.Poisson))*(uE.x + vE.y + wE.z)); 
	wSigma.z = (params.Young / ( 1 + params.Poisson))*(wE.z + (params.Poisson / ( 1 - 2*params.Poisson))*(uE.x + vE.y + wE.z));
	
	uSigma.y = vSigma.x = (params.Young / (1 + params.Poisson))*uE.y; //uE.y == vE.x
	uSigma.z = wSigma.x = (params.Young / (1 + params.Poisson))*uE.z;
	vSigma.z = wSigma.y = (params.Young / (1 + params.Poisson))*vE.z;		

    if (startIndex != 0xffffffff) {               
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {             
	            float3 referencePos_j = make_float3(FETCH(oldReferencePos, j));
				float4 measure = FETCH(oldMeasures, j);				
				float volume_j = measure.y;
				float tempExpr = 0.0f;
				float3 relPos = referencePos_i - referencePos_j;

				float dist = length(relPos);
				if (dist < params.smoothingRadius) {
					tempExpr =  (params.smoothingRadius - dist);					
					d.x = volume_j * params.SpikyKern * (relPos.x / dist) * tempExpr * tempExpr;
					d.y = volume_j * params.SpikyKern * (relPos.y / dist) * tempExpr * tempExpr;
					d.z = volume_j * params.SpikyKern * (relPos.z / dist) * tempExpr * tempExpr;

					tmpForce.x += -volume_i * dot(uSigma,d);
					tmpForce.y += -volume_i * dot(vSigma,d);
					tmpForce.z += -volume_i * dot(wSigma,d);					
				}                
            }
        }
    }
	return tmpForce;
}

__global__ void calcAccelerationD(
						  float4* acceleration,
						  float4* oldPos,	
						  float4* oldReferencePos,	
						  float4* olduDisplacementGradient,	
						  float4* oldvDisplacementGradient,	
						  float4* oldwDisplacementGradient,	
						  float4* oldMeasures,
						  uint* Index, 
						  uint* cellStart,
						  uint* cellEnd,
						  uint numParticles)			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    

	float3 pos = make_float3(FETCH(oldPos, index));
	float3 referencePos = make_float3(FETCH(oldReferencePos, index));
	float3 du_i = make_float3(FETCH(olduDisplacementGradient, index));
	float3 dv_i = make_float3(FETCH(oldvDisplacementGradient, index));
	float3 dw_i = make_float3(FETCH(oldwDisplacementGradient, index));
	float volume_i = FETCH(oldMeasures, index).y;

    int3 gridPos = calcGridPos(pos);
	float3 force = make_float3(0.0f);
	int cellcount = 2;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
				//!!! -=
                force -= sumForcePart(neighbourPos, index, referencePos, oldReferencePos, du_i, dv_i, dw_i, volume_i, oldMeasures, cellStart, cellEnd);
            }
        }
    }
    
	/*float3 temp = -1 * make_float3(FETCH(olduDisplacementGradient, index));
	uint originalIndex = Index[index];					*/

	uint originalIndex = Index[index];
	float3 acc = force / params.particleMass;	
	//float3 acc = make_float3(0.0f);
	acceleration[originalIndex] =  make_float4(acc, 0.0f);
}

__global__ void integrate(float4* posArray, //input / output 
						  float4* velArray, //input / output
						  float4* accArray, //input
						  uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

	volatile float4 posData = posArray[index];
	volatile float4 velData = velArray[index];	
	volatile float4 accData = accArray[index];	
	
	float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);
	float3 acc = make_float3(accData.x, accData.y, accData.z);

	vel += (params.gravity + acc) * params.deltaTime * velData.w; //don't integrate left plane particles, see initGrid function
	//vel += acc * params.deltaTime;   	
    pos += vel * params.deltaTime;  

	posArray[index] = make_float4(pos, posData.w);
	velArray[index] = make_float4(vel, velData.w);
}
#endif
