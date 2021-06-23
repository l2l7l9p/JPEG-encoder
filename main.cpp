#include"JPEGenc.h"
#include <string>

JPEGencoder a;

void test_cpu() {
	string loc = "data/";
    for (int i = 55000; i < 57000; ++ i) {
		string path = to_string(i);
        a.read_graph(loc + path + ".png");

        float cost_time = a.encode_cpu();

        printf("encode the %d graph cost %f ms\n", i, cost_time);
    }
    return ;
}

int main() {
	test_cpu();
	return 0;
}