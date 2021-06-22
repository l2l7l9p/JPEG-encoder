#include<cmath>
#include"JPEGenc.h"
using namespace std;

const float pi=acos(-1);

float R[MCUSIZE][MCUSIZE],G[MCUSIZE][MCUSIZE],B[MCUSIZE][MCUSIZE];

float Y[4][BLOCKSIZE][BLOCKSIZE],Cb[BLOCKSIZE][BLOCKSIZE],Cr[BLOCKSIZE][BLOCKSIZE];
void extract_YCbCr_and_subsample(const float *T,const float *bias) {
	for(int i=0; i<MCUSIZE; i++)
		for(int j=0; j<MCUSIZE; j++) {
			Y[(i>>3)*2+(j>>3)][i&7][j&7]=(T[0]*R[i][j]+T[1]*G[i][j]+T[2]*B[i][j])+bias[0];
			if (!(i&1) && !(j&1)) {
				Cb[i>>1][j>>1]=(T[3]*R[i][j]+T[4]*G[i][j]+T[5]*B[i][j])+bias[1];
				Cr[i>>1][j>>1]=(T[6]*R[i][j]+T[7]*G[i][j]+T[8]*B[i][j])+bias[2];
			}
		}
}

int F[BLOCKSIZE][BLOCKSIZE];
void DCT_and_quantize(float img[][BLOCKSIZE], const char *qTable) {
	for(int u=0; u<BLOCKSIZE; u++)
		for(int v=0; v<BLOCKSIZE; v++) {
			float Ff=0;
			for(int x=0; x<BLOCKSIZE; x++)
				for(int y=0; y<BLOCKSIZE; y++)
					Ff+=cos((2*x+1)*u*pi/16)*cos((2*x+1)*u*pi/16)*img[x][y];
				float Cu=(u==0) ?sqrt(2)/4 :0.5 ;
				float Cv=(v==0) ?sqrt(2)/4 :0.5 ;
				Ff*=Cu*Cv;
				F[u][v]=round(Ff/qTable[(u<<3)|v]);
		}
}

inline int get_size(int x) {
	int re=0;
	for(x=abs(x); x; x>>=1) re++;
	return re;
}

int codes[MAXCODELEN][2], codelen;
void huffman_coding(int val1,int val2,const int *huffTable) {
	++codelen;
	int val2size=get_size(val2);
	val1=huffTable[(val1<<4)|val2size];
	int val1size=get_size(val1)-1;
	codes[codelen][0]=val1size+val2size;
	codes[codelen][1]=((huffTable[val1]^(1<<val1size))<<val2size)+(val2>=0 ?val2 :(val2-1)&((1<<val2size)-1));
}

int zz[64];
void AC_coding(const char *zigzag,const int *AC_T) {
	for(int x=0; x<BLOCKSIZE; x++)
		for(int y=0; y<BLOCKSIZE; y++) zz[zigzag[(x<<3)|y]]=F[x][y];
	int zero=0;
	for(int k=1; k<64; k++) if (zz[k]==0) {
		zero++;
	} else {
		for(; zero>15; zero-=16) huffman_coding(15,0,AC_T);
		huffman_coding((zero<<4)|get_size(zz[k]),zz[k],AC_T);
		zero=0;
	}
	if (zero) huffman_coding(0,0,AC_T);
}

void encode_block(int &lastDC,float img[][BLOCKSIZE],const char *qTable,const char *zigzag,const int *DC_T,const int *AC_T) {
	// DCT and quantize
	DCT_and_quantize(img,qTable);
	// DC coding
	huffman_coding(0,F[0][0]-lastDC,DC_T);
	lastDC=F[0][0];
	// AC coding
	AC_coding(zigzag,AC_T);
}

float JPEGencoder::encode_cpu() {
	int lastYDC=0, lastCbDC=0, lastCrDC=0;
	codelen=0;
	
	for(int i=0; i<n; i+=MCUSIZE)
		for(int j=0; j<m; j+=MCUSIZE) {		// for each MCU :
			// extract RGB
			for(int x=0; x<MCUSIZE; x++)
				for(int y=0; y<MCUSIZE; y++) if (i+x<n && j+y<m) {
					R[x][y]=graph[i+x][j+y][0];
					G[x][y]=graph[i+x][j+y][1];
					B[x][y]=graph[i+x][j+y][2];
				} else R[x][y]=G[x][y]=B[x][y]=0;
			// extract YCbCr and subsample
			extract_YCbCr_and_subsample(T,bias);
			// for each block: DCT, quantize, DC coding, AC coding
			for(int yid=0; yid<4; yid++) encode_block(lastYDC,Y[yid],lq,zigzag,Y_DC_T,Y_AC_T);
			encode_block(lastCbDC,Cb,cq,zigzag,C_DC_T,C_AC_T);
			encode_block(lastCrDC,Cr,cq,zigzag,C_DC_T,C_AC_T);
		}
}