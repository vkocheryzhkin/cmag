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


struct Matrix3x4
{
  float a11,a12,a13,a14;
  float a21,a22,a23,a24;
  float a31,a32,a33,a34;
};

static __inline__ __device__ Matrix3x4 make_Matrix3x4()
{
  Matrix3x4 t; 
  t.a11 = 0; t.a12= 0; t.a13 = 0; t.a14 = 0;
  t.a21 = 0; t.a22= 0; t.a23 = 0; t.a24 = 0;
  t.a31 = 0; t.a32= 0; t.a33 = 0; t.a34 = 0;
  return t;
};
__device__ Matrix3x4 operator+ (const Matrix3x4 & a, const Matrix3x4 & b) 
{ 
	Matrix3x4 r;
	r.a11 = a.a11 + b.a11;
	r.a11 = a.a12 + b.a12;
	r.a11 = a.a13 + b.a13;
	r.a21 = a.a21 + b.a21;
	r.a22 = a.a22 + b.a22;
	r.a23 = a.a23 + b.a23;
	r.a31 = a.a31 + b.a31;
	r.a32 = a.a32 + b.a32;
	r.a33 = a.a33 + b.a33;
	return r; 
}
__device__ Matrix3x4 operator- (const Matrix3x4 & a, const Matrix3x4 & b) 
{ 
	Matrix3x4 r;
	r.a11 = a.a11 - b.a11;
	r.a11 = a.a12 - b.a12;
	r.a11 = a.a13 - b.a13;
	r.a21 = a.a21 - b.a21;
	r.a22 = a.a22 - b.a22;
	r.a23 = a.a23 - b.a23;
	r.a31 = a.a31 - b.a31;
	r.a32 = a.a32 - b.a32;
	r.a33 = a.a33 - b.a33;
	return r; 
}

__device__ Matrix3x4 operator+= (Matrix3x4 & a, const Matrix3x4 & b) 
{ 	
	a.a11 += b.a11;
	a.a12 += b.a12;
	a.a13 += b.a13;

	a.a21 += b.a21;
	a.a22 += b.a22;
	a.a23 += b.a23;

	a.a31 += b.a31;
	a.a32 += b.a32;
	a.a33 += b.a33;
	return a; 
}

__device__ Matrix3x4 operator* (const Matrix3x4 & a, const Matrix3x4 & b) 
{ 
	Matrix3x4 r;
	r.a11 = a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31;
	r.a12 = a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32;
	r.a13 = a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33;

	r.a21 = a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31;
	r.a22 = a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32;
	r.a23 = a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33;

	r.a31 = a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31;
	r.a32 = a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32;
	r.a33 = a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33;
	return r; 
}

__device__ float3 operator* (const Matrix3x4 & a, const float3 & b) 
{ 
	float3 r;
	r.x = a.a11 * b.x + a.a12 * b.y + a.a13 * b.z;	
	r.y = a.a21 * b.x + a.a22 * b.y + a.a23 * b.z;	
	r.z = a.a31 * b.x + a.a32 * b.y + a.a33 * b.z;	
	return r; 
}

__device__ Matrix3x4 operator* (const float & a, const Matrix3x4 & b) 
{ 
	Matrix3x4 r;
	r.a11 = a * b.a11;
	r.a12 = a * b.a12;
	r.a13 = a * b.a13;

	r.a21 = a * b.a21;
	r.a22 = a * b.a22;
	r.a23 = a * b.a23;

	r.a31 = a * b.a31;
	r.a32 = a * b.a32;
	r.a33 = a * b.a33;

	return r; 
}

__device__ Matrix3x4 Transpose (const Matrix3x4 & b) 
{ 
	Matrix3x4 r;
	r.a11 = b.a11;
	r.a12 = b.a21;
	r.a13 = b.a31;

	r.a21 = b.a12;
	r.a22 = b.a22;
	r.a23 = b.a32;

	r.a31 = b.a13;
	r.a32 = b.a23;
	r.a33 = b.a33;
	return r; 
}

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
	int cellcount = 1;
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

__device__ Matrix3x4 sumDisplacementGradientPart(
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
	Matrix3x4 gradient = make_Matrix3x4();	
	
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
					gradient.a11 += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * (relPos.x / dist) * tempExpr * tempExpr;
					gradient.a12 += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * (relPos.y / dist) * tempExpr * tempExpr;
					gradient.a13 += volume_j * (pos_j.x - pos_i.x - (referencePos_j.x - referencePos_i.x)) * params.SpikyKern * (relPos.z / dist) * tempExpr * tempExpr;
					
					gradient.a21 += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * (relPos.x / dist) * tempExpr * tempExpr;
					gradient.a22 += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * (relPos.y / dist) * tempExpr * tempExpr;
					gradient.a23 += volume_j * (pos_j.y - pos_i.y - (referencePos_j.y - referencePos_i.y)) * params.SpikyKern * (relPos.z / dist) * tempExpr * tempExpr;

					gradient.a31 += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * (relPos.x / dist) * tempExpr * tempExpr;
					gradient.a32 += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * (relPos.y / dist) * tempExpr * tempExpr;
					gradient.a33 += volume_j * (pos_j.z - pos_i.z - (referencePos_j.z - referencePos_i.z)) * params.SpikyKern * (relPos.z / dist) * tempExpr * tempExpr;																				
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
	Matrix3x4 result = make_Matrix3x4();	
	Matrix3x4 buf = make_Matrix3x4();	
	int cellcount = 1;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                buf = sumDisplacementGradientPart(neighbourPos, index, pos, oldPos, referencePos, oldReferencePos, oldMeasures, cellStart, cellEnd);
				result.a11 += buf.a11;
				result.a12 += buf.a12;
				result.a13 += buf.a13;

				result.a21 += buf.a21;
				result.a22 += buf.a22;
				result.a23 += buf.a23;

				result.a31 += buf.a31;
				result.a32 += buf.a32;
				result.a33 += buf.a33;
            }
        }
    }    				
	udisplacementGradient[index].x = result.a11;
	udisplacementGradient[index].y = result.a12;
	udisplacementGradient[index].z = result.a13;

	vdisplacementGradient[index].x = result.a21;
	vdisplacementGradient[index].y = result.a22;
	vdisplacementGradient[index].z = result.a23;

	wdisplacementGradient[index].x = result.a31;
	wdisplacementGradient[index].y = result.a32;	
	wdisplacementGradient[index].z = result.a33;
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
	Matrix3x4 Sigma = make_Matrix3x4();
	float3 d = make_float3(0.0f);	
		
	Matrix3x4 I = make_Matrix3x4();
	I.a11 = 1; I.a22 = 1; I.a33 = 1;		
	Matrix3x4 dUT = make_Matrix3x4();
	Matrix3x4 J = make_Matrix3x4();	

	dUT.a11 = du_i.x;
	dUT.a12 = du_i.y;
	dUT.a13 = du_i.z;

	dUT.a21 = dv_i.x;
	dUT.a22 = dv_i.y;
	dUT.a23 = dv_i.z;

	dUT.a31 = dw_i.x;
	dUT.a32 = dw_i.y;
	dUT.a33 = dw_i.z;	

	J = I + dUT;		
	

	//Green-Saint-Venant strain tensor	
	Matrix3x4 E = 0.5 * ( (Transpose(J)*J) - I);	

	float t1 = E.a11;// - (E.a11 + E.a22 + E.a33)/3;
	float t2 = E.a22;// - (E.a11 + E.a22 + E.a33)/3;
	float t3 = E.a33;// - (E.a11 + E.a22 + E.a33)/3;

	//Stress tensor
	Sigma.a11 = (params.Young / ( 1 + params.Poisson))*(t1 + (params.Poisson / ( 1 - 2 * params.Poisson))*(E.a11 + E.a22 + E.a33));
	Sigma.a22 = (params.Young / ( 1 + params.Poisson))*(t2 + (params.Poisson / ( 1 - 2 * params.Poisson))*(E.a11 + E.a22 + E.a33)); 
	Sigma.a33 = (params.Young / ( 1 + params.Poisson))*(t3 + (params.Poisson / ( 1 - 2 * params.Poisson))*(E.a11 + E.a22 + E.a33));
	
	Sigma.a12 = Sigma.a21 = (params.Young / (1 + params.Poisson))*E.a12;
	Sigma.a13 = Sigma.a31 = (params.Young / (1 + params.Poisson))*E.a13;
	Sigma.a23 = Sigma.a32 = (params.Young / (1 + params.Poisson))*E.a23;		

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

					tmpForce += -volume_i * (((I + dUT) * Sigma) * d);
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
	int cellcount = 1;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
				//!!! -=
                force -= sumForcePart(neighbourPos, index, referencePos, oldReferencePos, du_i, dv_i, dw_i, volume_i, oldMeasures, cellStart, cellEnd);
            }
        }
    }    	
	uint originalIndex = Index[index];
	float3 acc = force / params.particleMass;			
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
    pos += vel * params.deltaTime;  

	posArray[index] = make_float4(pos, posData.w);
	velArray[index] = make_float4(vel, velData.w);
}
#endif
