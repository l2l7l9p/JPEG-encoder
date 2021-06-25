main: main.cpp JPEGenc_cpu.cpp JPEGenc_gpu.cu JPEGenc_utils.cpp
	nvcc main.cpp JPEGenc_cpu.cpp JPEGenc_gpu.cu JPEGenc_utils.cpp -o main