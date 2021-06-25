#include"JPEGenc.h"

const float pi=acos(-1);

#define CHECK(call) {														\
	const cudaError_t error = call;											\
	if (error != cudaSuccess) {												\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s \n",error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}

float C_h[BLOCKSIZE*BLOCKSIZE];
void init_cos_h() {
	for(int x=0; x<BLOCKSIZE; x++)
		for(int u=0; u<BLOCKSIZE; u++) C_h[(x<<3)|u]=cos((2*x+1)*u*pi/16);
}

UC *G_d;
float *T_d,*bias_d,*C_d;
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
	CHECK(cudaMalloc(&C_d,sizeof(float)*64));
	CHECK(cudaMalloc(&zigzag_d,sizeof(char)*64));
	CHECK(cudaMalloc(&lq_d,sizeof(char)*64));
	CHECK(cudaMalloc(&cq_d,sizeof(char)*64));
	CHECK(cudaMalloc(&resY_d,sizeof(int)*fullsize));
	CHECK(cudaMalloc(&resCb_d,sizeof(int)*(fullsize>>2)));
	CHECK(cudaMalloc(&resCr_d,sizeof(int)*(fullsize>>2)));
	
	CHECK(cudaMemcpy(G_d,graph,sizeof(UC)*matrixSize,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(T_d,T,sizeof(float)*9,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(bias_d,bias,sizeof(float)*3,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(C_d,C_h,sizeof(float)*64,cudaMemcpyHostToDevice));
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
	CHECK(cudaFree(C_d));
	CHECK(cudaFree(bias_d));
	CHECK(cudaFree(zigzag_d));
	CHECK(cudaFree(lq_d));
	CHECK(cudaFree(cq_d));
	CHECK(cudaFree(resY_d));
	CHECK(cudaFree(resCb_d));
	CHECK(cudaFree(resCr_d));
}

#define DCT_and_quantize(Y,C,qTable,zigzag) {\
	float Ff=0;\
	for(int x=0; x<BLOCKSIZE; x++)\
		for(int y=0; y<BLOCKSIZE; y++)\
			Ff+=C[(x<<3)|bx]*C[(y<<3)|by]*Y[(tx&8)|x][(ty&8)|y];\
	Ff*=(bx==0) ?sqrtf(2)/4 :0.5 ;\
	Ff*=(by==0) ?sqrtf(2)/4 :0.5 ;\
	F[(yid<<6)+zigzag[bl]]=round(Ff/qTable[bl]);\
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
	float *T_d, float *bias_d, float *C_d,
	char *zigzag_d, char *lq_d, char *cq_d,
	int *resY_d, int *resCb_d, int *resCr_d
) {
	int tx=threadIdx.y, ty=threadIdx.x, bx=tx&7, by=ty&7, bl=(bx<<3)|by;
	int mx=blockDim.y*blockIdx.y+tx, my=blockDim.x*blockIdx.x+ty;
	int yid=((tx>>3)<<1)|(ty>>3);
	int MCUid=blockIdx.y*((m+15)>>4)+blockIdx.x;
	
	__shared__ float Y[MCUSIZE][MCUSIZE],Cb[MCUSIZE][MCUSIZE],Cr[MCUSIZE][MCUSIZE];
	__shared__ int F[MCUSIZE*MCUSIZE],flag[MCUSIZE*MCUSIZE];
	
	// extract YCbCr and subsample
	int id=mx*m+my;
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
	DCT_and_quantize(Y,C_d,lq_d,zigzag_d);
	// Y run-length coding
	scan(MCUSIZE);
	if ((id&63)==0 || F[id]!=0) resY_d[(MCUid<<8)|flag[id]]=(F[id]<<7)|bl;

	// Cb DCT and quantize
	DCT_and_quantize(Cb,C_d,cq_d,zigzag_d);
	// Cb run-length coding
	scan(BLOCKSIZE);
	if (id<64 && (id==0 || F[id]!=0)) resCb_d[(MCUid<<6)|flag[id]]=(F[id]<<7)|bl;
	
	// Cr DCT and quantize
	DCT_and_quantize(Cr,C_d,cq_d,zigzag_d);
	// Cr run-length coding
	scan(BLOCKSIZE);
	if (id<64 && (id==0 || F[id]!=0)) resCr_d[(MCUid<<6)|flag[id]]=(F[id]<<7)|bl;
}

__global__ void kernel_smem(
	int n,int m,
	UC *G_d,
	float *T_d, float *bias_d, float *C_d,
	char *zigzag_d, char *lq_d, char *cq_d,
	int *resY_d, int *resCb_d, int *resCr_d
) {
	int tx=threadIdx.y, ty=threadIdx.x, bx=tx&7, by=ty&7, bl=(bx<<3)|by;
	int mx=blockDim.y*blockIdx.y+tx, my=blockDim.x*blockIdx.x+ty;
	int yid=((tx>>3)<<1)|(ty>>3);
	int MCUid=blockIdx.y*((m+15)>>4)+blockIdx.x;
	
	__shared__ float Y[MCUSIZE][MCUSIZE],Cb[MCUSIZE][MCUSIZE],Cr[MCUSIZE][MCUSIZE];
	__shared__ int F[MCUSIZE*MCUSIZE],flag[MCUSIZE*MCUSIZE];
	__shared__ float T_s[9],bias_s[3],C_s[64];
	__shared__ char zigzag_s[64],lq_s[64],cq_s[64];
	int id=(tx<<4)|ty;
	if (id<9) T_s[id]=T_d[id];
	if (id<3) bias_s[id]=bias_d[id];
	if (id<64) {
		C_s[id]=C_d[id];
		zigzag_s[id]=zigzag_d[id];
		lq_s[id]=lq_d[id];
		cq_s[id]=cq_d[id];
	}
	
	// extract YCbCr and subsample
	id=mx*m+my;
	if (mx<n && my<m) {
		Y[tx][ty]=(T_s[0]*G_d[id*3]+T_s[1]*G_d[id*3+1]+T_s[2]*G_d[id*3+2])+bias_s[0];
		if (!(tx&1) && !(ty&1)) {
			Cb[tx>>1][ty>>1]=(T_s[3]*G_d[id]+T_s[4]*G_d[id*3+1]+T_s[5]*G_d[id*3+2])+bias_s[1];
			Cr[tx>>1][ty>>1]=(T_s[6]*G_d[id]+T_s[7]*G_d[id*3+1]+T_s[8]*G_d[id*3+2])+bias_s[2];
		}
	} else {
		Y[tx][ty]=0;
		if (!(tx&1) && !(ty&1)) Cb[tx>>1][ty>>1]=Cr[tx>>1][ty>>1]=0;
	}
	__syncthreads();
	
	id=(yid<<6)|bl;
	// Y DCT and quantize
	DCT_and_quantize(Y,C_s,lq_s,zigzag_s);
	// Y run-length coding
	scan(MCUSIZE);
	if ((id&63)==0 || F[id]!=0) resY_d[(MCUid<<8)|flag[id]]=(F[id]<<7)|bl;

	// Cb DCT and quantize
	DCT_and_quantize(Cb,C_s,cq_s,zigzag_s);
	// Cb run-length coding
	scan(BLOCKSIZE);
	if (id<64 && (id==0 || F[id]!=0)) resCb_d[(MCUid<<6)|flag[id]]=(F[id]<<7)|bl;
	
	// Cr DCT and quantize
	DCT_and_quantize(Cr,C_s,cq_s,zigzag_s);
	// Cr run-length coding
	scan(BLOCKSIZE);
	if (id<64 && (id==0 || F[id]!=0)) resCr_d[(MCUid<<6)|flag[id]]=(F[id]<<7)|bl;
}

#define DCT_mat_and_quantize(Y,C,qTable,zigzag) {\
	float Ff=0;\
	for(int x=0; x<BLOCKSIZE; x++)\
		Ff+=C[(x<<3)|bx]*Y[(tx&8)|x][(ty&8)|by];\
	F[(yid<<6)|bl]=Ff;\
	__syncthreads();\
	Ff=0;\
	for(int y=0; y<BLOCKSIZE; y++)\
		Ff+=C[(y<<3)|by]*F[(yid<<6)|(bx<<3)|y];\
	__syncthreads();\
	Ff*=(bx==0) ?sqrtf(2)/4 :0.5 ;\
	Ff*=(by==0) ?sqrtf(2)/4 :0.5 ;\
	F[(yid<<6)+zigzag[bl]]=round(Ff/qTable[bl]);\
	__syncthreads();\
}

__global__ void kernel_rearrange(
	int n,int m,
	UC *G_d,
	float *T_d, float *bias_d, float *C_d,
	char *zigzag_d, char *lq_d, char *cq_d,
	int *resY_d, int *resCb_d, int *resCr_d
) {
	int id=(threadIdx.y<<4)|threadIdx.x;
	int yid=id>>6;
	int bl=id&63, bx=bl>>3, by=bl&7;
	int tx=bx+((yid>>1)<<3), ty=by+((yid&1)<<3);
	int mx=blockDim.y*blockIdx.y+tx, my=blockDim.x*blockIdx.x+ty;
	int MCUid=blockIdx.y*((m+15)>>4)+blockIdx.x;
	
	__shared__ float Y[MCUSIZE][MCUSIZE],C[MCUSIZE][MCUSIZE];
	__shared__ int F[MCUSIZE*MCUSIZE],flag[MCUSIZE*MCUSIZE];
	
	// extract YCbCr and subsample
	id=mx*m+my;
	if (mx<n && my<m) {
		Y[tx][ty]=(T_d[0]*G_d[id*3]+T_d[1]*G_d[id*3+1]+T_d[2]*G_d[id*3+2])+bias_d[0];
		if (!(tx&1) && !(ty&1)) {
			C[tx>>1][ty>>1]=(T_d[3]*G_d[id]+T_d[4]*G_d[id*3+1]+T_d[5]*G_d[id*3+2])+bias_d[1];
			C[tx>>1][8+(ty>>1)]=(T_d[6]*G_d[id]+T_d[7]*G_d[id*3+1]+T_d[8]*G_d[id*3+2])+bias_d[2];
		}
	} else {
		Y[tx][ty]=0;
		if (!(tx&1) && !(ty&1)) C[tx>>1][ty>>1]=C[tx>>1][8+(ty>>1)]=0;
	}
	__syncthreads();
	
	id=(yid<<6)|bl;
	// Y DCT and quantize
	DCT_mat_and_quantize(Y,C_d,lq_d,zigzag_d);
	// Y run-length coding
	scan(MCUSIZE);
	if ((id&63)==0 || F[id]!=0) resY_d[(MCUid<<8)|flag[id]]=(F[id]<<7)|bl;

	// Cb&Cr DCT and quantize
	DCT_mat_and_quantize(C,C_d,cq_d,zigzag_d);
	// Cb&Cr run-length coding
	scan(BLOCKSIZE);
	if (id<64 && (id==0 || F[id]!=0)) resCb_d[(MCUid<<6)|flag[id]]=(F[id]<<7)|bl;
	if (64<=id && id<128 && (id==64 || F[id]!=0)) resCr_d[(MCUid<<6)|(flag[id]-flag[64])]=(F[id]<<7)|bl;
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
	clock_t t1=clock();
	init_cos_h();
	trans_from_host_to_device(n,m,graph,T,bias,zigzag,lq,cq);
	clock_t t2=clock();
	
	cudaEvent_t startTime, endTime;
	cudaEventCreate(&startTime);
	cudaEventCreate(&endTime);
	cudaEventRecord(startTime);
	dim3 grid((m+15)>>4,(n+15)>>4,1);
	dim3 block(16,16,1);
	// kernel<<<grid,block>>>(n,m,G_d,T_d,bias_d,C_d,zigzag_d,lq_d,cq_d,resY_d,resCb_d,resCr_d);
	// kernel_smem<<<grid,block>>>(n,m,G_d,T_d,bias_d,C_d,zigzag_d,lq_d,cq_d,resY_d,resCb_d,resCr_d);
	kernel_rearrange<<<grid,block>>>(n,m,G_d,T_d,bias_d,C_d,zigzag_d,lq_d,cq_d,resY_d,resCb_d,resCr_d);
	cudaEventRecord(endTime);
	cudaEventSynchronize(endTime);
	float duration=0;
	cudaEventElapsedTime(&duration,startTime,endTime);
	
	clock_t t3=clock();
	
	trans_from_device_to_host();
	
	codelen=0;
	int MCUnum=grid.x*grid.y;
	int lastYDC=0, lastCbDC=0, lastCrDC=0;
	for(int i=0; i<MCUnum; i++) {
		encode_block(resY,(i<<8),((i+1)<<8),lastYDC,Y_DC_T,Y_AC_T,this);
		encode_block(resCb,(i<<6),((i+1)<<6),lastCbDC,C_DC_T,C_AC_T,this);
		encode_block(resCr,(i<<6),((i+1)<<6),lastCrDC,C_DC_T,C_AC_T,this);
	}
	
	clock_t t4=clock();
	
	return duration+(t2-t1+t4-t3)/CLOCKS_PER_SEC*1000;
}