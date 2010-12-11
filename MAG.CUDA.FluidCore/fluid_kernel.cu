#ifndef _FLUID_KERNEL_CU_
#define _FLUID_KERNEL_CU_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "fluid_kernel.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float4, 1, cudaReadModeElementType> oldMeasuresTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);        
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint*   gridParticleHash,  // output
               uint*   gridParticleIndex, // output
               float4* pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint*   cellStart,        // output: cell start index
							      uint*   cellEnd,          // output: cell end index
							      float4* sortedPos,        // output: sorted positions
  							      float4* sortedVel,        // output: sorted velocities								  
                                  uint *  gridParticleHash, // input: sorted grid hashes
                                  uint *  gridParticleIndex,// input: sorted particle indices
				                  float4* oldPos,           // input: sorted position array
							      float4* oldVel,           // input: sorted velocity array
							      uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
    uint hash;
    // handle case when no. of particles not multiple of block size
    if (index < numParticles) {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look 
        // at neighboring particle's hash value without loading
        // two hash values per thread
	    sharedHash[threadIdx.x+1] = hash;

	    if (index > 0 && threadIdx.x == 0)
	    {
		    // first thread in block must load neighbor particle hash
		    sharedHash[0] = gridParticleHash[index-1];
	    }
	}

	__syncthreads();
	
	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

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

	    // Now use the sorted index to reorder the pos and vel data
	    uint sortedIndex = gridParticleIndex[index];
	    float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
        float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh

        sortedPos[index] = pos;
        sortedVel[index] = vel;
	}


}

__device__
float sumParticlesInDomain(int3    gridPos,
                   uint    index,
                   float3  pos,
                   float4* oldPos, 
                   uint*   cellStart,
                   uint*   cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    uint startIndex = FETCH(cellStart, gridHash);

    float sum = 0.0f;
    if (startIndex != 0xffffffff) {        // cell is not empty
        // iterate over particles in this cell
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
void calcDensityAndPressureD(			
			float4* measures, //output
			float4* oldPos,	 //input sorted position		
			uint* gridParticleIndex,
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
                sum += sumParticlesInDomain(neighbourPos, index, pos, oldPos, cellStart, cellEnd);
            }
        }
    }	
	float dens = sum * params.particleMass * params.Poly6Kern;
    measures[index].x = dens;	
	measures[index].y = (dens - params.restDensity) * params.gasConstant; 	
}
///////////////////////////////////////////////sumSpikyKernel

__device__
float3 sumNavierStokesForces(int3    gridPos,
                   uint    index,
                   float3  pos,
                   float4* oldPos, 
				   float3  vel,
				   float4* oldVel,
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
	            float3 pos2 = make_float3(FETCH(oldPos, j));
				float3 vel2 = make_float3(FETCH(oldVel, j));
				//place it inside branch
				float4 measure = FETCH(oldMeasures, j);
				float density2 = measure.x;
				float pressure2 = measure.y;				
				float tempExpr = 0.0f;

				float3 relPos = pos - pos2;
				float dist = length(relPos);

				if (dist < params.smoothingRadius) {
					tempExpr =  (params.smoothingRadius - dist);					
					pexp = pressure + pressure2;
					texp = tempExpr / density2;
					//todo: check minus for SpikyKern it looks like wrong operation
					tmpForce.x += texp * (params.SpikyKern * (relPos.x / dist) * tempExpr * pexp + params.LapKern * (vel2.x - vel.x));
					tmpForce.y += texp * (params.SpikyKern * (relPos.y / dist) * tempExpr * pexp + params.LapKern * (vel2.y - vel.y));
					tmpForce.z += texp * (params.SpikyKern * (relPos.z / dist) * tempExpr * pexp + params.LapKern * (vel2.z - vel.z));					
				}                
            }
        }
    }
	return tmpForce;				
}

///////////////////////////////////////////////calcAndApplyAccelerationD
__global__
void calcAndApplyAccelerationD(
			float4* acceleration,			
			float4* oldMeasures,
			float4* oldPos,			
			float4* oldVel,
			uint* gridParticleIndex,
			uint* cellStart,
			uint* cellEnd,
			uint numParticles)			
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    

	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(oldVel, index));
	float4 measure = FETCH(oldMeasures,index);
	float density = measure.x;
	float pressure = measure.y;

    int3 gridPos = calcGridPos(pos);

    float3 nvForce = make_float3(0.0f);
	int cellcount = 2;
    for(int z=-cellcount; z<=cellcount; z++) {
        for(int y=-cellcount; y<=cellcount; y++) {
            for(int x=-cellcount; x<=cellcount; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                nvForce += sumNavierStokesForces(neighbourPos, 
					index, 
					pos, 
					oldPos,
					vel,
					oldVel,
					pressure,					
					oldMeasures,
					cellStart, 
					cellEnd);
            }
        }
    }
	uint originalIndex = gridParticleIndex[index];					

	float3 acc =   params.particleMass * nvForce / density;
	float speed = dot(acc,acc);
	if(speed > params.accelerationLimit * params.accelerationLimit)
		acc *= params.accelerationLimit / sqrt(speed);

	acceleration[originalIndex] =  make_float4(acc, 0.0f);
}

// integrate particle attributes
__global__
void integrate(float4* posArray,  // input/output
               float4* velArray,  // input/output  
			   float4* velLeapFrogArray, // output
			   float4* acceleration, // input
               uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

	volatile float4 posData = posArray[index];    // ensure coalesced read
    volatile float4 velData = velArray[index];
	volatile float4 accData = acceleration[index];
	volatile float4 velLeapFrogData = velLeapFrogArray[index];

    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);
	float3 acc = make_float3(accData.x, accData.y, accData.z);

	float3 nextVel = vel + (params.gravity + acc) * params.deltaTime;

	float3 velLeapFrog = vel + nextVel;
	velLeapFrog *= 0.5;

    vel = nextVel;   	
    pos += vel * params.deltaTime;   

	float xb =(25 * 2)/64.0f -1.0f;
	//float xb = 0.0f;
	if (pos.x > xb - params.particleRadius) { pos.x = xb - params.particleRadius; vel.x *= params.boundaryDamping; }
    if (pos.x < -1.0f + params.particleRadius) { pos.x = -1.0f + params.particleRadius; vel.x *= params.boundaryDamping;}
    if (pos.y > 1.0f - params.particleRadius) { pos.y = 1.0f - params.particleRadius; vel.y *= params.boundaryDamping; }    
    if (pos.z > 1.0f - params.particleRadius) { pos.z = 1.0f - params.particleRadius; vel.z *= params.boundaryDamping; }
    if (pos.z < -1.0f + params.particleRadius) { pos.z = -1.0f + params.particleRadius; vel.z *= params.boundaryDamping;}
    if (pos.y < -1.0f + params.particleRadius) { pos.y = -1.0f + params.particleRadius; vel.y *= params.boundaryDamping;}		

    // store new position and velocity
    posArray[index] = make_float4(pos, posData.w);
    velArray[index] = make_float4(vel, velData.w);
	velLeapFrogArray[index] = make_float4(velLeapFrog, velLeapFrogData.w);
}
#endif
