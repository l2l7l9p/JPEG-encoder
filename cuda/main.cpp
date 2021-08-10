#include"JPEGenc.h"
#include <string>

JPEGencoder a;

void test_cpu() {
	string loc = "data/";
    for (int i = 55000; i < 55011; ++ i) {
		string path = to_string(i);
        a.read_graph(loc + path + ".png");

        float cost_time = a.encode_cpu();

        printf("encode the %d graph cost %f ms\n", i, cost_time);
    }
    return ;
}

void test_gpu() {
    string loc = "data/";
    for (int i = 55000; i < 55011; ++ i) {
		string path = to_string(i);
        a.read_graph(loc + path + ".png");

        float cost_time = a.encode_gpu();

        printf("encode the %d graph cost %f ms\n", i, cost_time);
    }
    return ;
}

int main(int argc,char *argv[]) {
	if (atoi(argv[1])==0) test_cpu();
		else test_gpu();
	return 0;
}