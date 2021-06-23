#include <cstdlib>
#include <ctime>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include"JPEGenc.h"

void JPEGencoder::read_graph(string path)	// read RGB graph from <path> to graph[][][3]
{
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count > 0) { // cuda
        srand(time(NULL));
        for (int i = 0; i < MAXN; ++ i){
            for (int j = 0; j < MAXM; ++ j){
                for (int k = 0; k < 3; ++ k){
                    graph[i][j][k] = rand() % 256;
                    data[i * MAXN * MAXM + j * MAXM + k] = rand() % 256;
                }
            }
        }
    } else { // cpu
        int channel;
        unsigned char *tmpData = stbi_load(path, &n, &m, &channel, 0);
        for (int i = 0 ; i < n; ++ i){
            for (int j = 0; j < m; ++ j){
                for (int k = 0; k < 3; ++ k){
                    data[i * n * m + j * m + k] = tmpData[i * n * m + j * m + k];
                    graph[i][j][k] = tmpData[i * n * m + j * m + k];
                }
            }
        }
        stbi_image_free(tmpData);
    }
}