#ifndef _FLUIDBEAM_KERNEL_CU_
#define _FLUIDBEAM_KERNEL_CU_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "fluidbeam_kernel.cuh"
#include "mag_math.h"

#if USE_TEX
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float4, 1, cudaReadModeElementType> oldDisplacementTex;										    
texture<float4, 1, cudaReadModeElementType> olduDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldvDisplacementGradientTex;
texture<float4, 1, cudaReadModeElementType> oldwDisplacementGradientTex;
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
void reorderDataAndFindCellStartD(uint*   cellStart,        // output
							      uint*   cellEnd,          // output
							      float4* sortedPos,        // output
  							      float4* sortedVel,        // output
								  float4* sortedDisplacement,//output
                                  uint*  gridParticleHash,  // input
                                  uint*  gridParticleIndex, // input
				                  float4* oldPos,           // input
							      float4* oldVel,           // input
								  float4* oldDisplacement,  //input
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
		float4 disp = FETCH(oldDisplacement, sortedIndex);

        sortedPos[index] = pos;
        sortedVel[index] = vel;
		sortedDisplacement[index] = disp;
	}
}

__device__
float sumParticlesInDomain(int3    gridPos,
                   uint    index,
                   float4  pos,				   
                   float4* oldPos, 
				   float4  vel,
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
	            float4 pos2 = FETCH(oldPos, j);
				float4 vel2 = FETCH(oldVel, j);				
				float3 relPos = make_float3(pos2 - pos); 
				float dist = length(relPos);

				 if(pos2.w == 0.0f)//refactor: escape beams density calculation
					 continue;

				if (dist < params.smoothingRadius) {					
					float wpolyExpr = pow(params.smoothingRadius,2)- pow(dist,2);
					sum += pow(wpolyExpr,3);																	
				}                
            }
        }
    }
    return sum;
}

__global__ 
void calcDensityAndPressureD(			
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

	float4 pos = FETCH(oldPos, index);
	float4 vel = FETCH(oldVel, index);

    int3 gridPos = calcGridPos(make_float3(pos));

    float sum = 0.0f;
	
    for(int z=-params.cellcount; z<=params.cellcount; z++) {
        for(int y=-params.cellcount; y<=params.cellcount; y++) {
            for(int x=-params.cellcount; x<=params.cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                sum += sumParticlesInDomain(neighbourPos, index, pos, oldPos, vel, oldVel, cellStart, cellEnd);
            }
        }
    }		
	float dens =  sum * params.particleMass * params.Poly6Kern;
    measures[index].x = dens;	
	measures[index].y = params.B * (pow(dens / params.restDensity ,7.0f) - 1.0f); 		
}

__device__ Matrix sumDisplacementGradient(
				   int3    gridPos,
                   uint    index,
                   float4  pos_i,
                   float4* oldPos, 
				   float4  disp_i,
                   float4* oldDisplacement, 				   
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
	            float4 pos_j = FETCH(oldPos, j);				
				float4 disp_j = FETCH(oldDisplacement,j);
				float volume_j = params.particleMass / 1000.0f;

				if(pos_j.w == 1.0f)//refactor: escape fluid displacement calculation
					 continue;

				float3 relPos = make_float3((pos_i - disp_i) - (pos_j - disp_j));
				float dist = length(relPos);				
				if (dist < params.smoothingRadius) {				
					float tempExpr = sinf((dist + params.smoothingRadius) * CUDART_PI_F / (2 * params.smoothingRadius));			
							
					gradient.a11 += volume_j * (disp_j.x - disp_i.x) * params.c * tempExpr * (relPos.x / dist);
					gradient.a12 += volume_j * (disp_j.x - disp_i.x) * params.c * tempExpr * (relPos.y / dist);
					gradient.a13 += volume_j * (disp_j.x - disp_i.x) * params.c * tempExpr * (relPos.z / dist);
					
					gradient.a21 += volume_j * (disp_j.y - disp_i.y) * params.c * tempExpr * (relPos.x / dist);
					gradient.a22 += volume_j * (disp_j.y - disp_i.y) * params.c * tempExpr * (relPos.y / dist);
					gradient.a23 += volume_j * (disp_j.y - disp_i.y) * params.c * tempExpr * (relPos.z / dist);

					gradient.a31 += volume_j * (disp_j.z - disp_i.z) * params.c * tempExpr * (relPos.x / dist);
					gradient.a32 += volume_j * (disp_j.z - disp_i.z) * params.c * tempExpr * (relPos.y / dist);
					gradient.a33 += volume_j * (disp_j.z - disp_i.z) * params.c * tempExpr * (relPos.z / dist);																				
				}                
            }
        }
    }
	return gradient;		
}

__global__ void calcDisplacementGradientD(										  
	  float4* uDisplacementGradient,
	  float4* vDisplacementGradient,
	  float4* wDisplacementGradient,
      float4* oldPos,                                          
	  float4* oldDisplacement, 
      uint* gridParticleIndex, //todo: remove
      uint* cellStart,
      uint* cellEnd,
      uint numParticles)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;  

	float4 pos = FETCH(oldPos, index);
	float4 disp = FETCH(oldDisplacement, index);

	int3 gridPos = calcGridPos(make_float3(pos));
	Matrix buf = make_Matrix();		
    for(int z= -params.cellcount; z<=params.cellcount; z++) {
        for(int y=-params.cellcount; y<=params.cellcount; y++) {
            for(int x=-params.cellcount; x<=params.cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                buf += sumDisplacementGradient(
					neighbourPos,
					index,
					pos,
					oldPos,
					disp,
					oldDisplacement,
					cellStart,
					cellEnd);				
            }
        }
    }    				
	uDisplacementGradient[index].x = buf.a11;
	uDisplacementGradient[index].y = buf.a12;
	uDisplacementGradient[index].z = buf.a13;

	vDisplacementGradient[index].x = buf.a21;
	vDisplacementGradient[index].y = buf.a22;
	vDisplacementGradient[index].z = buf.a23;

	wDisplacementGradient[index].x = buf.a31;
	wDisplacementGradient[index].y = buf.a32;	
	wDisplacementGradient[index].z = buf.a33;
}

__device__
float3 sumNavierStokesForces(int3    gridPos,
                   uint    index,
                   float4  pos,
                   float4* oldPos, 
				   float4  vel,
				   float4* oldVel,
				   float4 disp,
				   float4* oldDisplacement,
				   float4* olduDisplacementGradient,
				   float4* oldvDisplacementGradient,
				   float4* oldwDisplacementGradient,
				   float density,
				   float pressure,				   
				   float4* oldMeasures,
                   uint*   cellStart,
                   uint*   cellEnd)
{
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
				float4 disp2 = FETCH(oldDisplacement, j);		
				float4 measure = FETCH(oldMeasures, j);
				float density2 = measure.x;
				float pressure2 = measure.y;								
				float3 relPos = make_float3(pos - pos2);
				float dist = length(relPos);
				float artViscosity = 0.0f;
				float volume_i = params.particleMass / 1000.0f;
				float volume_j = params.particleMass / 1000.0f;
				
				if(pos.w != pos2.w){//different particles				  		
					float q = dist / params.particleRadius;
					float k = pow(params.soundspeed, 2);
					float3 coeff = k * (params.particleMass + params.particleMass) / params.particleMass 
						/ dist * normalize(relPos);					
					if((q >= 1.0f) && (q < 2.0f))
						tmpForce += coeff * 0.5f * pow(2.0f - q, 2);
					if((q >= 2.0f / 3.0f) && (q < 1.0f))
						tmpForce += coeff * (2.0f * q - 3.0f / 2 * q * q);
					if(q < 2.0f / 3.0f)
						tmpForce += coeff * 2.0f / 3.0f;	
					 continue;
				 }
				
				 if(pos.w == 1.0f && pos2.w ==1.0f){//fluid + fluid
					if (dist < params.smoothingRadius) {
						float temp = (params.smoothingRadius - dist);				
						float artViscosity = 0.0f;
						float vij_pij = dot(make_float3(vel - vel2),relPos);
						if(vij_pij < 0){						
							float nu = 2.0f * 0.18f * params.smoothingRadius *
								params.soundspeed / (density + density2);

							artViscosity = -1.0f * nu * vij_pij / 
								(dot(relPos, relPos) + 0.01f * pow(params.smoothingRadius, 2));
						}
						tmpForce +=  -1.0f * params.particleMass *
							(pressure / pow(density,2) + pressure2 / pow(density2,2) +
							artViscosity) * params.SpikyKern * normalize(relPos) * temp * temp;							
					}    
				}					
				//pos == j
				//if(pos.w == 0.0f && pos2.w ==0.0f){//beam						
				//	relPos = make_float3((pos2 - disp2) - (pos - disp));
				//	dist = length(relPos);			
				//	if (dist < params.smoothingRadius) {					
				//		float tempExpr = sinf((dist + params.smoothingRadius) *
				//			CUDART_PI_F / (2 * params.smoothingRadius));	
				//		Matrix Stress = make_Matrix();
				//		float3 d = make_float3(0.0f);			
				//		Matrix I = make_Matrix();
				//		I.a11 = 1; I.a22 = 1; I.a33 = 1;		
				//		Matrix dU = make_Matrix();
				//		Matrix J = make_Matrix();	
				//		Matrix E = make_Matrix();	

				//		float3 du_i = make_float3(FETCH(olduDisplacementGradient, j));
				//		float3 dv_i = make_float3(FETCH(oldvDisplacementGradient, j));
				//		float3 dw_i = make_float3(FETCH(oldwDisplacementGradient, j));

				//		dU.a11 = du_i.x; dU.a12 = du_i.y; dU.a13 = du_i.z;
				//		dU.a21 = dv_i.x; dU.a22 = dv_i.y; dU.a23 = dv_i.z;
				//		dU.a31 = dw_i.x; dU.a32 = dw_i.y; dU.a33 = dw_i.z;
				//		J = I + Transpose(dU);										
				//		//Green-Saint-Venant strain tensor	
				//		E = 0.5 * ((Transpose(J)*J) - I);	
				//		float a11 = params.Young / ((1 + params.Poisson) * (1 - 2 * params.Poisson));
				//		float a12 = params.Young / (1 + params.Poisson);
				//		//Stress tensor					
				//		Stress.a11 = a11 * ((1 - params.Poisson) * E.a11 + params.Poisson * (E.a22 + E.a33));
				//		Stress.a22 = a11 * ((1 - params.Poisson) * E.a22 + params.Poisson * (E.a11 + E.a33));
				//		Stress.a33 = a11 * ((1 - params.Poisson) * E.a33 + params.Poisson * (E.a11 + E.a22));
				//		
				//		Stress.a12 = Stress.a21 = a12 * E.a12;
				//		Stress.a13 = Stress.a31 = a12 * E.a13;
				//		Stress.a23 = Stress.a32 = a12 * E.a23;

				//		d.x = volume_j * params.c * (relPos.x / dist) * tempExpr;
				//		d.y = volume_j * params.c * (relPos.y / dist) * tempExpr;
				//		d.z = volume_j * params.c * (relPos.z / dist) * tempExpr;																		

				//		tmpForce += -volume_i * (((I + Transpose(dU)) * Stress) * d);					
				//	} 
				//}
            }
        }
    }		
	return tmpForce;
}

__global__ void calcAndApplyAccelerationD(
			float4* acceleration,	
			float4* olduDisplacementGradient,
			float4* oldvDisplacementGradient,
			float4* oldwDisplacementGradient,
			float4* oldMeasures,
			float4* oldPos,			
			float4* oldVel,
			float4* oldDisplacement,
			uint* gridParticleIndex,
			uint* cellStart,
			uint* cellEnd,
			uint numParticles)			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    

	float4 pos = FETCH(oldPos, index);
	float4 vel = FETCH(oldVel, index);
	float4 disp = FETCH(oldDisplacement, index);
	float4 measure = FETCH(oldMeasures,index);
	float density = measure.x;
	float pressure = measure.y;

    int3 gridPos = calcGridPos(make_float3(pos));

    float3 force = make_float3(0.0f);	
    for(int z=-params.cellcount; z<=params.cellcount; z++) {
        for(int y=-params.cellcount; y<=params.cellcount; y++) {
            for(int x=-params.cellcount; x<=params.cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);				
					force += sumNavierStokesForces(
						neighbourPos, 
						index, 
						pos, 
						oldPos,
						vel,
						oldVel,
						disp,
						oldDisplacement,
						olduDisplacementGradient,
						oldvDisplacementGradient,
						oldwDisplacementGradient,		
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
	acceleration[originalIndex] =  make_float4(acc, 0.0f);
}

__global__
void integrate(float4* posArray,		 // input, output
               float4* velArray,		 // input, output  
			   float4* displacementArray, //input, output
			   float4* velLeapFrogArray, // output
			   float4* acceleration,	 // input
               uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          

	volatile float4 posData = posArray[index]; 
    volatile float4 velData = velArray[index];
	volatile float4 dispData = displacementArray[index];
	volatile float4 accData = acceleration[index];
	volatile float4 velLeapFrogData = velLeapFrogArray[index];

    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);
	float3 disp = make_float3(dispData.x, dispData.y, dispData.z);
	float3 acc = make_float3(accData.x, accData.y, accData.z);

	//float3 nextVel = vel + (params.gravity + acc) * params.deltaTime;
	float3 nextVel = vel + (params.gravity + acc) * params.deltaTime * velData.w; //todo: remove w usage
	

	float3 velLeapFrog = vel + nextVel;
	velLeapFrog *= 0.5;

    vel = nextVel;   
	disp += vel * params.deltaTime;
    pos += vel * params.deltaTime;   

	float scale = params.gridSize.x * params.particleRadius;
	float bound = 2.0f * params.particleRadius * params.fluidParticlesSize.z - 1.0f * scale;	
	float offset = params.boundaryOffset * 2 * params.particleRadius;
	//float bound = 2.0f * params.particleRadius * (params.fluidParticlesSize.z + 6) - 1.0f * scale;		

	if(posData.w == 1.0f){
		/*if (pos.x > 1.0f * scale - offset - params.particleRadius) {
			pos.x = 1.0f * scale - offset - params.particleRadius; vel.x *= params.boundaryDamping; }*/
		if (pos.x < -1.0f * scale + offset + params.particleRadius) {
			pos.x = -1.0f * scale + offset + params.particleRadius; vel.x *= params.boundaryDamping;}
		if (pos.y < -1.0f * scale + offset + params.particleRadius) {
			pos.y = -1.0f * scale + offset + params.particleRadius; vel.y *= params.boundaryDamping;}	
		/*if (pos.y > 1.0f * scale - offset -  params.particleRadius) {
			pos.y = 1.0f * scale - offset - params.particleRadius; vel.y *= params.boundaryDamping; }    
		if (pos.z > bound - offset - params.particleRadius) {
			pos.z = bound - offset - params.particleRadius; vel.z *= params.boundaryDamping; }
		if (pos.z < -1.0f * scale + offset + params.particleRadius) {
			pos.z = -1.0f * scale + offset + params.particleRadius; vel.z *= params.boundaryDamping;}
		*/	
	}
    
    posArray[index] = make_float4(pos, posData.w);
	displacementArray[index] = make_float4(disp, dispData.w);
    velArray[index] = make_float4(vel, velData.w);
	velLeapFrogArray[index] = make_float4(velLeapFrog, velLeapFrogData.w);
}
#endif
