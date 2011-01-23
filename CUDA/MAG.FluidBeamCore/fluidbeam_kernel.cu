#ifndef _FLUIDBEAM_KERNEL_CU_
#define _FLUIDBEAM_KERNEL_CU_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "mag_math.h"
#include "math_constants.h"
#include "fluidbeam_kernel.cuh"

#if USE_TEX
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldReferencePosTex;
texture<float4, 1, cudaReadModeElementType> olduDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldvDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldwDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float4, 1, cudaReadModeElementType> oldMeasuresTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif
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

__global__
void calcHashD(uint*   gridParticleHash,  // output
               uint*   gridParticleIndex, // output
               float4* pos,               // input
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
    volatile float4 p = pos[index];

    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

__global__
void reorderDataAndFindCellStartD(uint*   cellStart,         // output
							      uint*   cellEnd,           // output
							      float4* sortedPos,         // output
								  float4* sortedReferencePos,// output
  							      float4* sortedVel,         // output
                                  uint *  gridParticleHash,  // input
                                  uint *  gridParticleIndex, // input
				                  float4* oldPos,            // input
								  float4* oldReferencePos,   // input
							      float4* oldVel,            // input
							      uint    numParticles)
{
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
		sortedReferencePos[index] = FETCH(oldReferencePos, sortedIndex);
        sortedVel[index] = vel;
	}


}

__device__ float sumDensity(
	int3 gridPos,
    uint index,
    float3 position,
    float4* oldPos, 
	float3 velocity,
	float4* oldVel, 
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
	            float3 positionJ = make_float3(FETCH(oldPos, j));
				float3 velocityJ = make_float3(FETCH(oldVel, j));
				float3 relPosition = position - positionJ; 				
				float dist = length(relPosition);
				if (dist < params.smoothingRadius) {
					/*float temp = params.smoothingRadius - dist;					
					sum += temp * temp * dot(velocity - velocityJ, normalize(relPosition));*/
					float wpolyExpr = pow(params.smoothingRadius,2)- pow(dist,2);					
					sum += pow(wpolyExpr,3);
				}                
            }
        }
    }
    return sum;
}

__global__ void CalculateDensityAndPressureD(			
			float4* measures, //output
			float4* oldPos,	  //input 
			float4* oldVel,	  //input 
			uint* gridParticleIndex,
			uint* cellStart,
			uint* cellEnd,
			uint numParticles)
			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    
	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(oldVel, index));
    int3 gridPos = calcGridPos(pos);

    float sum = 0.0f;	
    for(int z= -params.cellcount; z <= params.cellcount; z++) {
        for(int y= -params.cellcount; y <= params.cellcount; y++) {
            for(int x= -params.cellcount; x <= params.cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                sum += sumDensity(neighbourPos, index, pos, oldPos, vel, oldVel, cellStart, cellEnd);
            }
        }
    }
	float density = sum * params.particleMass * params.Poly6Kern;
	//float density = params.restDensity + params.deltaTime * sum * params.particleMass * params.SpikyKern;
	//float density = sum;
    measures[index].x = density;	
	measures[index].y = (density - params.restDensity) * params.gasConstant; 		
	//measures[index].y = params.B * (pow(density / params.restDensity ,7) - 1);
}

__device__ Matrix sumDisplacementGradient(
				   int3    gridPos,
                   uint    index,
                   float3  pos_i,
                   float4* oldPos, 
				   float3  referencePos_i,
                   float4* oldReferencePos, 				   
                   uint*   cellStart,
                   uint*   cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

    uint startIndex = FETCH(cellStart, gridHash);    	
	Matrix gradient = make_Matrix();	
	
    if (startIndex != 0xffffffff) {               
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {             
	            float3 pos_j = make_float3(FETCH(oldPos, j));				
				float3 referencePos_j = make_float3(FETCH(oldReferencePos,j));
				float volume_j = params.particleMass /  params.restDensity;

				float3 relPos = referencePos_i - referencePos_j;
				float distance = length(relPos);				
				if (distance < params.smoothingRadius) {				
					float temp = params.smoothingRadius - distance;						
					gradient.a11 += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * temp  * temp * (relPos.x / distance);
					gradient.a12 += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * temp  * temp * (relPos.y / distance);
					gradient.a13 += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * temp  * temp * (relPos.z / distance);
					
					gradient.a21 += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * temp * temp * (relPos.x / distance);
					gradient.a22 += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * temp * temp * (relPos.y / distance);
					gradient.a23 += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * temp * temp * (relPos.z / distance);

					gradient.a31 += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * temp * temp * (relPos.x / distance);
					gradient.a32 += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * temp * temp * (relPos.y / distance);
					gradient.a33 += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * temp * temp * (relPos.z / distance);																				
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
						  uint* Index, 
						  uint* cellStart,
						  uint* cellEnd,
						  uint numParticles)			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    

	float3 pos = make_float3(FETCH(oldPos, index));
	float3 referencePos = make_float3(FETCH(oldReferencePos, index));
    int3 gridPos = calcGridPos(referencePos);	
	Matrix buf = make_Matrix();	
	int cellcount = 1;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                buf += sumDisplacementGradient(neighbourPos, index, pos, oldPos, referencePos, oldReferencePos, cellStart, cellEnd);				
            }
        }
    }    				
	udisplacementGradient[index].x = buf.a11;
	udisplacementGradient[index].y = buf.a12;
	udisplacementGradient[index].z = buf.a13;

	vdisplacementGradient[index].x = buf.a21;
	vdisplacementGradient[index].y = buf.a22;
	vdisplacementGradient[index].z = buf.a23;

	wdisplacementGradient[index].x = buf.a31;
	wdisplacementGradient[index].y = buf.a32;	
	wdisplacementGradient[index].z = buf.a33;
}

__device__ float3 ElasticForce(
	float4  referencePos_i,
	float4  referencePos_j,
	uint j,
	float4*  olduDisplacementGradient,
    float4*  oldvDisplacementGradient,
    float4*  oldwDisplacementGradient)
{
	float volume_i = params.particleMass / params.restDensity;
	float volume_j = volume_i;
	float3 relPos = make_float3(referencePos_i - referencePos_j);
	float distance = length(relPos);
	if (distance < params.smoothingRadius) {					
		float temp = params.smoothingRadius - distance;
		Matrix Stress = make_Matrix();
		float3 d = make_float3(0.0f);			
		Matrix I = make_Matrix();
		I.a11 = 1; I.a22 = 1; I.a33 = 1;		
		Matrix dU = make_Matrix();
		Matrix J = make_Matrix();	
		Matrix E = make_Matrix();	
		float3 du_i = make_float3(FETCH(olduDisplacementGradient, j));
		float3 dv_i = make_float3(FETCH(oldvDisplacementGradient, j));
		float3 dw_i = make_float3(FETCH(oldwDisplacementGradient, j));
		dU.a11 = du_i.x; dU.a12 = du_i.y; dU.a13 = du_i.z;
		dU.a21 = dv_i.x; dU.a22 = dv_i.y; dU.a23 = dv_i.z;
		dU.a31 = dw_i.x; dU.a32 = dw_i.y; dU.a33 = dw_i.z;
		J = I + Transpose(dU);											
		//Green-Saint-Venant strain tensor	
		E = 0.5 * ((Transpose(J)*J) - I);	

		float a11 = params.Young / ((1 + params.Poisson) * (1 - 2 * params.Poisson));
		float a12 = params.Young / (1 + params.Poisson);

		//Stress tensor					
		Stress.a11 = a11 * ((1 - params.Poisson) * E.a11 + params.Poisson * (E.a22 + E.a33));
		Stress.a22 = a11 * ((1 - params.Poisson) * E.a22 + params.Poisson * (E.a11 + E.a33));
		Stress.a33 = a11 * ((1 - params.Poisson) * E.a33 + params.Poisson * (E.a11 + E.a22));
		
		Stress.a12 = Stress.a21 = a12 * E.a12;
		Stress.a13 = Stress.a31 = a12 * E.a13;
		Stress.a23 = Stress.a32 = a12 * E.a23;							
		d = volume_j * params.SpikyKern * temp * temp * normalize(relPos);
		return  -volume_i * (((I + Transpose(dU)) * Stress) * d);
	}
	return make_float3(0.0f);
}

//__device__ float3 NavierStokesForces(
//	uint j,
//    float4  pos,
//    float4 pos_j, 
//	float3  vel,
//	float4* oldVel,
//	float pressure,				   
//	float4* oldMeasures)    
//{           
//    float3 pos2 = make_float3(pos_j);
//	float3 vel2 = make_float3(FETCH(oldVel, j));				
//	float4 measure = FETCH(oldMeasures, j);
//	float density2 = measure.x;
//	float pressure2 = measure.y;					
//
//	float3 relPos = make_float3(pos) - pos2;
//	float dist = length(relPos);
//
//	if (dist < params.smoothingRadius) {		
//		float temp =  (params.smoothingRadius - dist);			
//		return -0.5f * (pressure + pressure2) / density2 * normalize(relPos) * params.SpikyKern * temp * temp;
//			 + params.LapKern * (vel2 - vel) * temp / density2;
//	}                          
//	return make_float3(0.0f);	
//}

__device__ float3 sumForce(
				   int3 gridPos,
                   uint index,
				   float4  pos,
                   float4* oldPos, 
				   float3  vel,
				   float4* oldVel,
				   float pressure,				   
				   float4* oldMeasures,
                   float4  referencePos_j,
                   float4* oldReferencePos, 
				   float4* olduDisplacementGradient,
				   float4* oldvDisplacementGradient,
				   float4* oldwDisplacementGradient,                   				   
                   uint* cellStart,
                   uint* cellEnd)
{
	uint gridHash = calcGridHash(gridPos);
    uint startIndex = FETCH(cellStart, gridHash);    
	float3 tmpForce = make_float3(0.0f);
	float texp = 0.0f;
	float pexp = 0.0f;
	/*float4 measure = FETCH(oldMeasures,index);
	float density = measure.x;	*/	

    if (startIndex != 0xffffffff) {               
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j = startIndex; j < endIndex; j++) {
            if (j != index) {         
					//float4 referencePos_i = FETCH(oldReferencePos, j);
					//float4 pos_j = FETCH(oldPos, j);					
					//if((referencePos_i.w == 1.0f) && (referencePos_j.w == 1.0f)){
					//	/*tmpForce +=	ElasticForce(
					//		referencePos_i,
					//		referencePos_j,
					//		j,
					//		olduDisplacementGradient,
					//		oldvDisplacementGradient,
					//		oldwDisplacementGradient);		*/	
					//	//tmpForce.x += -0.1f;
					//}
					//if((pos.w == -1.0f) && (pos_j.w == -1.0f)){
					//	tmpForce += params.particleMass / density * NavierStokesForces(
					//		j,
					//		pos,
					//		pos_j,
					//		vel,
					//		oldVel,
					//		pressure,
					//		oldMeasures
					//		);								
					//}	

					float3 pos2 = make_float3(FETCH(oldPos, j));
					float3 vel2 = make_float3(FETCH(oldVel, j));				
					float4 measure = FETCH(oldMeasures, j);
					float density2 = measure.x;
					float pressure2 = measure.y;				
					float tempExpr = 0.0f;

					float3 relPos = make_float3(pos) - pos2;
					float dist = length(relPos);

					if (dist < params.smoothingRadius) {
						tempExpr =  (params.smoothingRadius - dist);					
						pexp = pressure + pressure2;
						texp = tempExpr / density2;					
						tmpForce.x += texp * (-0.5 * params.SpikyKern * (relPos.x / dist) * tempExpr * pexp + params.LapKern * (vel2.x - vel.x));
						tmpForce.y += texp * (-0.5 * params.SpikyKern * (relPos.y / dist) * tempExpr * pexp + params.LapKern * (vel2.y - vel.y));
						tmpForce.z += texp * (-0.5 * params.SpikyKern * (relPos.z / dist) * tempExpr * pexp + params.LapKern * (vel2.z - vel.z));					
					}           
				}                
            }
        }    
	return tmpForce;
}


__global__
void calcAccelerationD(
			float4* acceleration,			
			float4* oldPos,	
			float4* oldReferencePos,	
			float4* olduDisplacementGradient,	
			float4* oldvDisplacementGradient,	
			float4* oldwDisplacementGradient,						
			float4* oldVel,
			float4* oldMeasures,
			uint* gridParticleIndex,
			uint* cellStart,
			uint* cellEnd,
			uint numParticles)			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    

	float4 pos = FETCH(oldPos, index);
	float3 vel = make_float3(FETCH(oldVel, index));
	float4 referencePos_j = FETCH(oldReferencePos, index);	
	float4 measure = FETCH(oldMeasures,index);
	float density = measure.x;
	float pressure = measure.y;	

    int3 gridPos = calcGridPos(make_float3(referencePos_j));
	float3 force = make_float3(0.0f);
	int cellcount = 1;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);				             
				force +=sumForce(
					neighbourPos,
					index, 
					pos,
					oldPos,
					vel,
					oldVel,
					pressure,					
					oldMeasures,
					referencePos_j,
					oldReferencePos,
					olduDisplacementGradient,
					oldvDisplacementGradient,
					oldwDisplacementGradient,					
					cellStart,
					cellEnd);				
            }
        }
    }    	
	uint originalIndex = gridParticleIndex[index];
	float3 acc = force;			
	
	float speed = dot(acc,acc);
	if(speed > params.accelerationLimit * params.accelerationLimit)
		acc *= params.accelerationLimit / sqrt(speed);

	acceleration[originalIndex] =  make_float4(acc, 0.0f);
	/*uint originalIndex = gridParticleIndex[index];
	float3 acc = force / params.particleMass;		
	acceleration[originalIndex] =  make_float4(acc, 0.0f);*/
}

__global__
void integrate(float4* posArray,		 // input, output
               float4* velArray,		 // input, output  
			   float4* velLeapFrogArray, // output
			   float4* acceleration,	 // input
               uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          

	volatile float4 posData = posArray[index]; 
    volatile float4 velData = velArray[index];
	volatile float4 accData = acceleration[index];
	volatile float4 velLeapFrogData = velLeapFrogArray[index];

    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);
	float3 acc = make_float3(accData.x, accData.y, accData.z);

	//float3 nextVel = vel + (params.gravity + acc) * params.deltaTime;
	//float3 nextVel = vel + (params.gravity + acc) * params.deltaTime * velData.w;
	float3 nextVel = vel + acc * params.deltaTime * velData.w;

	float3 velLeapFrog = vel + nextVel;
	velLeapFrog *= 0.5;

    vel = nextVel;   	
    pos += vel * params.deltaTime;   

	//float bound =(25 * 2)/64.0f -1.0f;			
	float bound =(25 * 2)/64.0f -1.0f;
	if (pos.x > 1.0f - params.particleRadius) { pos.x = 1.0f - params.particleRadius; vel.x *= params.boundaryDamping; }
    if (pos.x < -1.0f + params.particleRadius) { pos.x = -1.0f + params.particleRadius; vel.x *= params.boundaryDamping;}
    if (pos.y > 1.0f - params.particleRadius) { pos.y = 1.0f - params.particleRadius; vel.y *= params.boundaryDamping; }    
    if (pos.z > bound - params.particleRadius) { pos.z = bound - params.particleRadius; vel.z *= params.boundaryDamping; }
    if (pos.z < -1.0f + params.particleRadius) { pos.z = -1.0f + params.particleRadius; vel.z *= params.boundaryDamping;}
    if (pos.y < -1.0f + params.particleRadius) { pos.y = -1.0f + params.particleRadius; vel.y *= params.boundaryDamping;}		
	
	//float ybound = -2 * params.particleRadius;// -1.0f;	
	//if (pos.y > ybound) return;	 
    
    posArray[index] = make_float4(pos, posData.w);
    velArray[index] = make_float4(vel, velData.w);
	velLeapFrogArray[index] = make_float4(velLeapFrog, velLeapFrogData.w);
}
#endif
