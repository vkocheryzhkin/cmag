#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include "beamSystem.cuh"
#include "beam_kernel.cuh"


typedef unsigned int uint;

BOOST_AUTO_TEST_CASE(calcDisplacementGradientTest)
{
	 //BOOST_CHECK(true);
	cudaInit(1,(char **) &"");
	int numParticles = 2;
	uint numGridCells = 64*64*46;
	unsigned int memSize = sizeof(float) * 4 * numParticles; 

	SimParams params;
	params.particleRadius = 4.0f;
	setParameters(&params);

	//device
	float* dUdisplacementGradient;
	float* dVdisplacementGradient;
	float* dWdisplacementGradient;	
	float* dSortedPos;
	float* dSortedReferencePos;
	float* dSortedMeasures; 
	uint* dIndex;
	uint* dCellStart;
	uint* dCellEnd;

	allocateArray((void**)&dSortedPos, memSize);
	allocateArray((void**)&dUdisplacementGradient, memSize);
	allocateArray((void**)&dVdisplacementGradient, memSize);
	allocateArray((void**)&dWdisplacementGradient, memSize);
	allocateArray((void**)&dSortedReferencePos, memSize);
	allocateArray((void**)&dSortedMeasures, memSize);
	allocateArray((void**)&dIndex, numParticles*sizeof(uint));
	allocateArray((void**)&dCellStart, numGridCells*sizeof(uint));
	allocateArray((void**)&dCellEnd, numGridCells*sizeof(uint));	

	//host
	float *hSortedPos = new float[numParticles*4];
	float *hSortedReferencePos = new float[numParticles*4];
	float *hSortedMeasures = new float[numParticles*4];
	uint *hIndex = new uint[numParticles];	
	uint *hCellStart = new uint[numGridCells];	
	uint *hCellEnd =  new uint[numGridCells];			
	float *hResult = new float[numParticles*4];	//output

	memset(hSortedPos, 0, numParticles*4*sizeof(float));
	memset(hSortedReferencePos, 0, numParticles*4*sizeof(float));	
	memset(hSortedMeasures, 0, numParticles*4*sizeof(float));
	memset(hIndex, 0, numParticles*sizeof(uint));
	memset(hCellStart, 0, numGridCells*sizeof(uint));
	memset(hCellEnd, 0, numGridCells*sizeof(uint));
	memset(hResult, 0, numParticles*4*sizeof(float));
			
	copyArrayToDevice(dSortedPos, hSortedPos,0, numParticles*4*sizeof(float));
	copyArrayToDevice(dSortedReferencePos, hSortedReferencePos,0, numParticles*4*sizeof(float));
	copyArrayToDevice(dSortedMeasures, hSortedMeasures, 0, numParticles*4*sizeof(float));
	copyArrayToDevice(dIndex, hIndex, 0, numParticles*sizeof(uint));
	copyArrayToDevice(dCellStart, hCellStart, 0, numGridCells*sizeof(uint));
	copyArrayToDevice(dCellEnd, hCellEnd, 0, numGridCells*sizeof(uint));
	
	calcDisplacementGradient(	
		dUdisplacementGradient, 
		dVdisplacementGradient, 
		dWdisplacementGradient, 
		dSortedPos,	
		dSortedReferencePos,		
		dSortedMeasures,
		dIndex,
		dCellStart,
		dCellEnd,
		numParticles,
		numGridCells
	);
	
	copyArrayFromDevice(hResult, dUdisplacementGradient ,0 , numParticles*4*sizeof(float));

	for(uint i=0; i < numParticles; i++) 
	{
		printf("%f %f %f %f \n", 					
				hResult[4*i+0],
				hResult[4*i+1],	
				hResult[4*i+2],
				hResult[4*i+3]				
		);
	}

	
	delete [] hSortedPos; 
	delete [] hSortedReferencePos; 
	delete [] hSortedMeasures; 
	delete [] hIndex; 
	delete [] hCellStart;
	delete [] hCellEnd;
	delete [] hResult;

	freeArray(dSortedPos);
	freeArray(dUdisplacementGradient);
	freeArray(dVdisplacementGradient);
	freeArray(dWdisplacementGradient);	
	freeArray(dSortedReferencePos);
	freeArray(dSortedMeasures);
	freeArray(dIndex);
	freeArray(dCellStart);
	freeArray(dCellEnd);	
}
