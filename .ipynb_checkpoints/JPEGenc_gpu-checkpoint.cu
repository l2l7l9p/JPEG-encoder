#include"JPEGenc.h"

#define CHECK(call) {														\
	const cudaError_t error = call;											\
	if (error != cudaSuccess) {												\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s \n",error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}

UC *G_d;
float *T_d,*bias_d;
char *zigzag_d,*lq_d,*cq_d;
int *resY_d,*resCb_d,*resCr_d, fullsize;
void trans_from_host_to_device(
	int n,int m,
	UC *graph,
	const float *T, const float *bias, const char *zigzag,
	const char *lq, const char *cq
) {
	int matrixSize=n*m*3;
	fullsize=(((n+15)>>4)<<4)*(((m+15)>>4)<<4);
	CHECK(cudaMalloc(&G_d,sizeof(UC)*matrixSize));
	CHECK(cudaMalloc(&T_d,sizeof(float)*9));
	CHECK(cudaMalloc(&bias_d,sizeof(float)*3));
	CHECK(cudaMalloc(&zigzag_d,sizeof(char)*64));
	CHECK(cudaMalloc(&lq_d,sizeof(char)*64));
	CHECK(cudaMalloc(&cq_d,sizeof(char)*64));
	CHECK(cudaMalloc(&resY_d,sizeof(int)*fullsize));
	CHECK(cudaMalloc(&resCb_d,sizeof(int)*(fullsize>>2)));
	CHECK(cudaMalloc(&resCr_d,sizeof(int)*(fullsize>>2)));
	
	CHECK(cudaMemcpy(G_d,graph,sizeof(UC)*matrixSize,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(T_d,T,sizeof(float)*9,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(bias_d,bias,sizeof(float)*3,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(zigzag_d,zigzag,sizeof(char)*64,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(lq_d,lq,sizeof(char)*64,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(cq_d,cq,sizeof(char)*64,cudaMemcpyHostToDevice));
	CHECK(cudaMemset(resY_d,-1,sizeof(int)*fullsize));
	CHECK(cudaMemset(resCb_d,-1,sizeof(int)*(fullsize>>2)));
	CHECK(cudaMemset(resCr_d,-1,sizeof(int)*(fullsize>>2)));
}

int resY[MAXN*MAXM],resCb[MAXN*MAXM],resCr[MAXN*MAXM];
void trans_from_device_to_host() {
	CHECK(cudaMemcpy(resY,resY_d,sizeof(int)*fullsize,cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(resCb,resCb_d,sizeof(int)*(fullsize>>2),cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(resCr,resCr_d,sizeof(int)*(fullsize>>2),cudaMemcpyDeviceToHost));
	CHECK(cudaFree(G_d));
	CHECK(cudaFree(T_d));
	CHECK(cudaFree(bias_d));
	CHECK(cudaFree(zigzag_d));
	CHECK(cudaFree(lq_d));
	CHECK(cudaFree(cq_d));
	CHECK(cudaFree(resY_d));
	CHECK(cudaFree(resCb_d));
	CHECK(cudaFree(resCr_d));
}

#define DCT_and_quantize(Y,qTable) {\
	float Ff=0, pi=acosf(-1);\
	for(int x=0; x<BLOCKSIZE; x++)\
		for(int y=0; y<BLOCKSIZE; y++)\
			Ff+=cosf((2*x+1)*bx*pi/16)*cosf((2*y+1)*by*pi/16)*Y[(tx&8)|x][(ty&8)|y];\
	Ff*=(bx==0) ?sqrtf(2)/4 :0.5 ;\
	Ff*=(by==0) ?sqrtf(2)/4 :0.5 ;\
	F[(yid<<6)+zigzag_d[bl]]=round(Ff/qTable[bl]);\
	__syncthreads();\
}

#define scan(size) {\
	flag[id]=((id&63)==0 || F[id]!=0);\
	for(int s=1; s<size*size; s<<=1) {\
		__syncthreads();\
		int val=(id>=s) ?flag[id-s] :0 ;\
		__syncthreads();\
		flag[id]+=val;\
	}\
}

__global__ void kernel(
	int n,int m,
	UC *G_d,
	float *T_d, float *bias_d, char *zigzag_d,
	char *lq_d, char *cq_d,
	int *resY_d, int *resCb_d, int *resCr_d
) {
	int tx=threadIdx.y, ty=threadIdx.x, bx=tx&7, by=ty&7, bl=(bx<<3)|by;
	int mx=blockDim.y*blockIdx.y+tx, my=blockDim.x*blockIdx.x+ty;
	int id=mx*m+my;
	int yid=((tx>>3)<<1)|(ty>>3);
	int MCUid=blockIdx.y*((m+15)>>4)+blockIdx.x;
	
	__shared__ float Y[MCUSIZE][MCUSIZE],Cb[MCUSIZE][MCUSIZE],Cr[MCUSIZE][MCUSIZE];
	__shared__ int F[MCUSIZE*MCUSIZE],flag[MCUSIZE*MCUSIZE];
	
	// extract YCbCr and subsample
	if (mx<n && my<m) {
		Y[tx][ty]=(T_d[0]*G_d[id*3]+T_d[1]*G_d[id*3+1]+T_d[2]*G_d[id*3+2])+bias_d[0];
		if (!(tx&1) && !(ty&1)) {
			Cb[tx>>1][ty>>1]=(T_d[3]*G_d[id]+T_d[4]*G_d[id*3+1]+T_d[5]*G_d[id*3+2])+bias_d[1];
			Cr[tx>>1][ty>>1]=(T_d[6]*G_d[id]+T_d[7]*G_d[id*3+1]+T_d[8]*G_d[id*3+2])+bias_d[2];
		}
	} else {
		Y[tx][ty]=0;
		if (!(tx&1) && !(ty&1)) Cb[tx>>1][ty>>1]=Cr[tx>>1][ty>>1]=0;
	}
	__syncthreads();
	
	id=(yid<<6)|bl;
	// Y DCT and quantize
	DCT_and_quantize(Y,lq_d);
	// Y run-length coding
	scan(MCUSIZE);
	if ((id&63)==0 || F[id]!=0) resY_d[(MCUid<<8)|flag[id]]=(F[id]<<7)|bl;

	// Cb DCT and quantize
	DCT_and_quantize(Cb,cq_d);
	// Cb run-length coding
	scan(BLOCKSIZE);
	if (id<64 && (id==0 || F[id]!=0)) resCb_d[(MCUid<<6)|flag[id]]=(F[id]<<7)|bl;
	
	// Cr DCT and quantize
	DCT_and_quantize(Cr,cq_d);
	// Cr run-length coding
	scan(BLOCKSIZE);
	if (id<64 && (id==0 || F[id]!=0)) resCr_d[(MCUid<<6)|flag[id]]=(F[id]<<7)|bl;
}

void encode_block(int *res,int st,int en,int &lastDC,const int *DC_T, const int *AC_T,JPEGencoder *ts) {
	int lastPos=63;
	for(int j=st; j<en; j++) {
		if (res[j]&(1<<6)) break;
		int pos=res[j]&63, val=res[j]>>7;
		if (pos==0) {	// Y DC
			if (lastPos!=63) ts->huffman_coding(0,0,AC_T);
			ts->huffman_coding(0,val-lastDC,DC_T);
			lastDC=val;
		} else {		// Y AC
			int zero=pos-lastPos-1;
			for(; zero>15; zero-=16) ts->huffman_coding(15,0,AC_T);
			ts->huffman_coding(zero,val,AC_T);
		}
		lastPos=pos;
	}
	if (lastPos!=63) ts->huffman_coding(0,0,AC_T);
}

float JPEGencoder::encode_gpu() {
	cudaEvent_t startTime, endTime;
	cudaEventCreate(&startTime);
	cudaEventCreate(&endTime);
	cudaEventRecord(startTime);
	
	trans_from_host_to_device(n,m,graph,T,bias,zigzag,lq,cq);
	
	dim3 grid((m+15)>>4,(n+15)>>4,1);
	dim3 block(16,16,1);
	kernel<<<grid,block>>>(n,m,G_d,T_d,bias_d,zigzag_d,lq_d,cq_d,resY_d,resCb_d,resCr_d);
	cudaDeviceSynchronize();
	
	trans_from_device_to_host();
	
	codelen=0;
	int MCUnum=grid.x*grid.y;
	int lastYDC=0, lastCbDC=0, lastCrDC=0;
	for(int i=0; i<MCUnum; i++) {
		encode_block(resY,(i<<8),((i+1)<<8),lastYDC,Y_DC_T,Y_AC_T,this);
		encode_block(resCb,(i<<6),((i+1)<<6),lastCbDC,C_DC_T,C_AC_T,this);
		encode_block(resCr,(i<<6),((i+1)<<6),lastCrDC,C_DC_T,C_AC_T,this);
	}
	
	cudaEventRecord(endTime);
	cudaEventSynchronize(endTime);
	float duration=0;
	cudaEventElapsedTime(&duration,startTime,endTime);
	return duration;
}