**This is a record of two course projects. You may read the codes and functions for reference. But it could not be a release version.**

**这是两个课程 project，代码仅供参考，不是直接拿来跑的发行版**



reference book: *Ze-nian Li, Fundamentals of Multimedia, second version*



## python

Run it as `python main.py <graph_name>`. It will generate a JPEG graph by hand and a GIF graph by `matplotlib.image`, and then do some error estimation.

## cuda

I Implemented a class `JPEGencoder`. Call the function `read_graph(path)` to read a graph, and `encode_cpu()` or `encode_gpu()` to encode, which return the running time and discard the generated bit stream. (for the reason that I only concern about how cuda speeds it up)

`main.cpp` is for reference.