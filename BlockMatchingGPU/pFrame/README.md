full_search_multi_stream_without_dct.cu:
包含20个stream，15个GOP，共300 frame
只包含p frame的motion estimation和prediction，没有reconst（dct，qp，idct）
每个stream完成一个完整的GOP，先拿第一帧作为ref，第二帧curr跑一个kernel，再以predict（temp）作为ref，第2到14帧作为curr，跑另一个kernel 13次
