#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include "fluidbeam_kernel.cu"

extern "C"
{

void cudaInit(int argc, char **argv)
{   
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        cutilDeviceInit(argc, argv);
    } else {
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
    }
}

void cudaGLInit(int argc, char **argv)
{   
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        cutilDeviceInit(argc, argv);
    } else {
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    }
}

void allocateArray(void **devPtr, size_t size)
{
    cutilSafeCall(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
    cutilSafeCall(cudaFree(devPtr));
}

void threadSync()
{
    cutilSafeCall(cudaThreadSynchronize());
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
    cutilSafeCall(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, 
					       cudaGraphicsMapFlagsNone));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
    cutilSafeCall(cudaGraphicsUnregisterResource(cuda_vbo_resource));	
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
    void *ptr;
    cutilSafeCall(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
    size_t num_bytes; 
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,  
						       *cuda_vbo_resource));
    return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
   cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void copyArrayFromDevice(void* host, const void* device, 
			 struct cudaGraphicsResource **cuda_vbo_resource, int size)
{   
    if (cuda_vbo_resource)
	device = mapGLBufferObject(cuda_vbo_resource);

    cutilSafeCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    
    if (cuda_vbo_resource)
	unmapGLBufferObject(*cuda_vbo_resource);
}

void setParameters(SimParams *hostParams)
{
    cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
}

uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

void integrateSystem(float* pos,
                     float* vel,  
					 float* displacement,
					 float* velLeapFrog,
					 float* acc,
                     uint numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    integrate<<< numBlocks, numThreads >>>((float4*)pos,
                                           (float4*)vel,
										   (float4*)displacement,
										   (float4*)velLeapFrog,
										   (float4*)acc,
                                           numParticles);
    
    cutilCheckMsg("integrate kernel execution failed");
}

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                           gridParticleIndex,
                                           (float4 *) pos,
                                           numParticles);
    
    cutilCheckMsg("Kernel execution failed");
}

void reorderDataAndFindCellStart(uint*  cellStart,
							     uint*  cellEnd,
							     float* sortedPos,
							     float* sortedVel,
								 float* sortedDisplacement,
                                 uint*  gridParticleHash,
                                 uint*  gridParticleIndex,
							     float* oldPos,
							     float* oldVel,
								 float* oldDisplacement,
							     uint   numParticles,
							     uint   numCells)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

	cutilSafeCall(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));
	//cutilSafeCall(cudaMemset(cellEnd, 0xffffffff, numCells*sizeof(uint)));

	#if USE_TEX
		cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
		cutilSafeCall(cudaBindTexture(0, oldDisplacementTex, oldDisplacement, numParticles*sizeof(float4)));
	#endif

		uint smemSize = sizeof(uint)*(numThreads+1);
		reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
			cellStart,
			cellEnd,
			(float4 *) sortedPos,
			(float4 *) sortedVel,
			(float4 *) sortedDisplacement,
			gridParticleHash,
			gridParticleIndex,
			(float4 *) oldPos,
			(float4 *) oldVel,
			(float4 *) oldDisplacement,
			numParticles);
		cutilCheckMsg("Kernel execution failed: reorderDataAndFindCellStartD");

	#if USE_TEX
		cutilSafeCall(cudaUnbindTexture(oldPosTex));
		cutilSafeCall(cudaUnbindTexture(oldVelTex));
		cutilSafeCall(cudaUnbindTexture(oldDisplacementTex));
	#endif
}

void calcDensityAndPressure(			
			float* measures,
			float* sortedPos,			
			float* sortedVel,
			uint* gridParticleIndex,
			uint* cellStart,
			uint* cellEnd,
			uint numParticles,
			uint numGridCells)
{

	#if USE_TEX
    cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
    cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
    cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
	#endif

	uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    calcDensityAndPressureD<<< numBlocks, numThreads >>>(										  
										  (float4*)measures,
                                          (float4*)sortedPos,                                          
										  (float4*)sortedVel, 
                                          gridParticleIndex,
                                          cellStart,
                                          cellEnd,
                                          numParticles);

    cutilCheckMsg("Kernel execution failed");

	#if USE_TEX
    cutilSafeCall(cudaUnbindTexture(oldPosTex));
	cutilSafeCall(cudaUnbindTexture(oldVelTex));
    cutilSafeCall(cudaUnbindTexture(cellStartTex));
    cutilSafeCall(cudaUnbindTexture(cellEndTex));
	#endif
}

void calcDisplacementGradient(
	float* duDisplacementGradient,
	float* dvDisplacementGradient,
	float* dwDisplacementGradient,
	float* sortedPos,
	float* sortedDisplacement,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles,
	uint numGridCells)
{
	#if USE_TEX
    cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, oldDisplacementTex, sortedDisplacement, numParticles*sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
    cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
	#endif
	uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    calcDisplacementGradientD<<< numBlocks, numThreads >>>(										  
										  (float4*)duDisplacementGradient,
										  (float4*)dvDisplacementGradient,
										  (float4*)dwDisplacementGradient,
                                          (float4*)sortedPos,                                          
										  (float4*)sortedDisplacement, 
                                          gridParticleIndex,
                                          cellStart,
                                          cellEnd,
                                          numParticles);
	#if USE_TEX
    cutilSafeCall(cudaUnbindTexture(oldPosTex));    
	cutilSafeCall(cudaUnbindTexture(oldDisplacementTex)); 
	cutilSafeCall(cudaUnbindTexture(cellStartTex));
    cutilSafeCall(cudaUnbindTexture(cellEndTex));
	#endif
}


void calcAndApplyAcceleration(
	float* acceleration,
	float* duDisplacementGradient,
	float* dvDisplacementGradient,
	float* dwDisplacementGradient,
	float* sortedMeasures,			
	float* sortedPos,			
	float* sortedVel,
	float* sortedDisplacement,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles,
	uint numGridCells)
{
	#if USE_TEX
    cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
    cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, oldDisplacementTex, sortedDisplacement, numParticles*sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, olduDisplacementGradientTex, duDisplacementGradient, numParticles * sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, oldvDisplacementGradientTex, dvDisplacementGradient, numParticles * sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, oldwDisplacementGradientTex, dwDisplacementGradient, numParticles * sizeof(float4)));
	cutilSafeCall(cudaBindTexture(0, oldMeasuresTex, sortedMeasures, numParticles*sizeof(float4)));
    cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numGridCells*sizeof(uint)));
    cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numGridCells*sizeof(uint)));    
	#endif

	uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    calcAndApplyAccelerationD<<< numBlocks, numThreads >>>(
										  (float4*) acceleration,
										  (float4*) duDisplacementGradient,
										  (float4*) dvDisplacementGradient,
					  					  (float4*) dwDisplacementGradient,
										  (float4*) sortedMeasures,										  
                                          (float4*) sortedPos,                                          
										  (float4*) sortedVel, 
										  (float4*) sortedDisplacement,
                                          gridParticleIndex,
                                          cellStart,
                                          cellEnd,
                                          numParticles);

    cutilCheckMsg("Kernel execution failed");

	#if USE_TEX
    cutilSafeCall(cudaUnbindTexture(oldPosTex));
    cutilSafeCall(cudaUnbindTexture(oldVelTex));
	cutilSafeCall(cudaUnbindTexture(oldDisplacementTex));	
	cutilSafeCall(cudaUnbindTexture(olduDisplacementGradientTex));	
	cutilSafeCall(cudaUnbindTexture(oldvDisplacementGradientTex));	
	cutilSafeCall(cudaUnbindTexture(oldwDisplacementGradientTex));	
	cutilSafeCall(cudaUnbindTexture(oldMeasuresTex));
    cutilSafeCall(cudaUnbindTexture(cellStartTex));
    cutilSafeCall(cudaUnbindTexture(cellEndTex));
	#endif
}
}// extern "C"
