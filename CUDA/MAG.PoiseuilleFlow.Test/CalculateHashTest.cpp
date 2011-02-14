#include <boost/test/unit_test.hpp>
#include <vector_types.h>
#include <vector_functions.h>
typedef unsigned int uint;

#include "poiseuilleFlowSystem.cuh"
#include "magUtil.cuh"

//struct InitCuda
//{	
//	InitCuda() 
//	{
//		BOOST_TEST_MESSAGE("setup");		
//	}
//	~InitCuda()
//	{
//		BOOST_TEST_MESSAGE("teardown");
//	}
//};


BOOST_FIXTURE_TEST_SUITE(CalculateHash)//, InitCuda)

BOOST_AUTO_TEST_CASE(CalculateHash_ReturnOne)
{	
	
	const int numParticles = 1;
	SimParams params;
	params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	params.particleRadius = 1.0f / 64;
	params.gridSize = make_uint3(64, 64, 64);
	setParameters(&params); 

	float *hPosition, *dPosition;
	uint *dHash, *dIndex, *hHash;	

	hPosition = new float[numParticles * 4];
	hHash = new uint[numParticles];	
	hPosition[1] = hPosition[2] = params.worldOrigin.x + params.particleRadius;	
	hPosition[0] = params.worldOrigin.x + 2.0f * params.particleRadius;

	uint memSize = 4 * numParticles * sizeof(float);
	allocateArray((void**)&dHash,  numParticles * sizeof(uint));
	allocateArray((void**)&dIndex,  numParticles * sizeof(uint));
	allocateArray((void**)&dPosition,  memSize);
	copyArrayToDevice(dPosition, hPosition, 0, memSize);

	calcHash(dHash, dIndex, dPosition, numParticles);

	copyArrayFromDevice(hHash, dHash, 0, numParticles * sizeof(uint));

	BOOST_REQUIRE_EQUAL(1, hHash[0]);
	
	freeArray(dPosition); 	
	freeArray(dIndex); 
	freeArray(dHash); 		
	delete [] hPosition;
	delete [] hHash;
}

BOOST_AUTO_TEST_CASE(CalculateImageHash_ReturnOppositeHash)
{		
	const int numParticles = 1;
	SimParams params;
	params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	params.particleRadius = 1.0f / 64;
	params.gridSize = make_uint3(64, 64, 64);
	setParameters(&params); 

	float *hPosition, *dPosition;
	uint *dHash, *dIndex, *hHash;	

	hPosition = new float[numParticles * 4];
	hHash = new uint[numParticles];	
	hPosition[1] = hPosition[2] = params.worldOrigin.x + params.particleRadius;	
	hPosition[0] = params.worldOrigin.x - params.particleRadius;

	uint memSize = 4 * numParticles * sizeof(float);
	allocateArray((void**)&dHash,  numParticles * sizeof(uint));
	allocateArray((void**)&dIndex,  numParticles * sizeof(uint));
	allocateArray((void**)&dPosition,  memSize);
	copyArrayToDevice(dPosition, hPosition, 0, memSize);

	calcHash(dHash, dIndex, dPosition, numParticles);

	copyArrayFromDevice(hHash, dHash, 0, numParticles * sizeof(uint));

	BOOST_REQUIRE_EQUAL(params.gridSize.x - 1, hHash[0]);

	freeArray(dPosition); 	
	freeArray(dIndex); 
	freeArray(dHash); 		
	delete [] hPosition;
	delete [] hHash;
}


BOOST_AUTO_TEST_SUITE_END()
