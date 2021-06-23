#include<cmath>
#include"JPEGenc.h"
using namespace std;

const float pi=acos(-1);

UC R[MCUSIZE][MCUSIZE],G[MCUSIZE][MCUSIZE],B[MCUSIZE][MCUSIZE];

float Y[4][BLOCKSIZE][BLOCKSIZE],Cb[BLOCKSIZE][BLOCKSIZE],Cr[BLOCKSIZE][BLOCKSIZE];
void extract_YCbCr_and_subsample(const float *T,const float *bias) {
	for(int i=0; i<MCUSIZE; i++)
		for(int j=0; j<MCUSIZE; j++) {
			Y[(i>>3)*2+(j>>3)][i&7][j&7]=(T[0]*R[i][j]+T[1]*G[i][j]+T[2]*B[i][j])+bias[0];
			if (!(i&1) && !(j&1)) {
				Cb[(i>>1)&7][(j>>1)&7]=(T[3]*R[i][j]+T[4]*G[i][j]+T[5]*B[i][j])+bias[1];
				Cr[(i>>1)&7][(j>>1)&7]=(T[6]*R[i][j]+T[7]*G[i][j]+T[8]*B[i][j])+bias[2];
			}
		}
}

int F[BLOCKSIZE*BLOCKSIZE];
void DCT_and_quantize(float img[][BLOCKSIZE], const char *qTable, const char *zigzag) {
	for(int u=0; u<BLOCKSIZE; u++)
		for(int v=0; v<BLOCKSIZE; v++) {
			float Ff=0;
			for(int x=0; x<BLOCKSIZE; x++)
				for(int y=0; y<BLOCKSIZE; y++)
					Ff+=cos((2*x+1)*u*pi/16)*cos((2*x+1)*u*pi/16)*img[x][y];
			float Cu=(u==0) ?sqrt(2)/4 :0.5 ;
			float Cv=(v==0) ?sqrt(2)/4 :0.5 ;
			Ff*=Cu*Cv;
			F[zigzag[(u<<3)|v]]=round(Ff/qTable[(u<<3)|v]);
		}
}

void AC_coding(const int *AC_T,JPEGencoder *ts) {
	int zero=0;
	for(int k=1; k<64; k++) if (F[k]==0) {
		zero++;
	} else {
		for(; zero>15; zero-=16) ts->huffman_coding(15,0,AC_T);
		ts->huffman_coding((zero<<4)|get_size(F[k]),F[k],AC_T);
		zero=0;
	}
	if (zero) ts->huffman_coding(0,0,AC_T);
}

void encode_block(
	int &lastDC,
	float img[][BLOCKSIZE],
	const char *qTable, const char *zigzag,
	const int *DC_T, const int *AC_T,
	JPEGencoder *ts
) {
	// DCT and quantize
	DCT_and_quantize(img,qTable,zigzag);
	// DC coding
	ts->huffman_coding(0,F[0]-lastDC,DC_T);
	lastDC=F[0];
	// AC coding
	AC_coding(AC_T,ts);
}

float JPEGencoder::encode_cpu() {
	clock_t startTime=clock();
	
	int lastYDC=0, lastCbDC=0, lastCrDC=0;
	codelen=0;
	for(int i=0; i<n; i+=MCUSIZE)
		for(int j=0; j<m; j+=MCUSIZE) {		// for each MCU :
			// extract RGB
			for(int x=0; x<MCUSIZE; x++)
				for(int y=0; y<MCUSIZE; y++) if (i+x<n && j+y<m) {
					int gid=(i+x)*m+(j+y);
					R[x][y]=graph[gid*3];
					G[x][y]=graph[gid*3+1];
					B[x][y]=graph[gid*3+2];
				} else R[x][y]=G[x][y]=B[x][y]=0;
			// extract YCbCr and subsample
			extract_YCbCr_and_subsample(T,bias);
			// for each block: DCT, quantize, DC coding, AC coding
			for(int yid=0; yid<4; yid++) encode_block(lastYDC,Y[yid],lq,zigzag,Y_DC_T,Y_AC_T,this);
			encode_block(lastCbDC,Cb,cq,zigzag,C_DC_T,C_AC_T,this);
			encode_block(lastCrDC,Cr,cq,zigzag,C_DC_T,C_AC_T,this);
		}
	
	clock_t endTime=clock();
	
	return endTime-startTime;
}