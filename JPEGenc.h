#include<bits/stdc++.h>
using namespace std;

#ifndef HEADER
#define HEADER

const int MAXN=1e4, MAXM=1e4;

class JPEGencoder {
	public:
	
	int n,m;
	float graph[MAXN][MAXM][3];
	
	void read_graph(string path);
	
	bool encode();
}

#endif