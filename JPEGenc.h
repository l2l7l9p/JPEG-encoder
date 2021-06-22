#include<bits/stdc++.h>
using namespace std;

#ifndef HEADER
#define HEADER

class JPEGencoder {
	public:
	
	// *************** constants (definition in JPEGenc_const.cpp) *************
	
	#define MAXN 2000
	#define MAXM 2000
	#define MAXCODELEN 1000000
	#define MCUSIZE 16
	#define BLOCKSIZE 8
	static const char lq[64];
	static const char cq[64];
	static const char zigzag[64];
	static const int Y_DC_T[12];
	static const int Y_AC_T[256];
	static const int C_DC_T[12];
	static const int C_AC_T[256];
	static const float T[9];
	static const float bias[3];
	
	// *************** vars *************
	
	int n,m;
	float graph[MAXN][MAXM][3];
	
	// *************** constants *************
	
	void read_graph(string path);	// read RGB graph from <path> to graph[][][3]
	float encode_cpu();
	float encode_gpu();
};

#endif