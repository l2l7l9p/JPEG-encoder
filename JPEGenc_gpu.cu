#include"JPEGenc.h"

float JPEGencoder::encode_gpu() {
	cudaEvent_t startTime, endTime;
	cudaEventCreate(&startTime);
	cudaEventCreate(&endTime);
	cudaEventRecord(startTime);
	
	int r=8; // each block has 2^r threads
	int matrixSize=n*m;
	kernel_cntsm<<<((matrixSize+(1<<r)-1)>>r),(1<<r)>>>(mat_d,result_d,n,m);
	
	cudaEventRecord(endTime);
	cudaEventSynchronize(endTime);
	float duration=0;
	cudaEventElapsedTime(&duration,startTime,endTime);
	return duration;
}