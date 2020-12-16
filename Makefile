
RT_INC = /home/luojiapeng/Libs/TensorRT-7.2.1.6/include/
RT_LIB = /home/luojiapeng/Libs/TensorRT-7.2.1.6/lib/
OPENCV_INC = /usr/include/opencv4/
CUDA_INC = /usr/local/cuda-11.1/include/
CUDA_LIB = /usr/local/cuda-11.1/lib64/
OPENCV_LIB = /usr/lib/aarch64-linux-gnu/
DEBUG_FLAGS = -O3

BIN_DIR = ./bin

all: directories trt_onnx

directories:
	mkdir -p $(BIN_DIR)

trt_onnx: trt_onnx.cpp *.hpp
	g++ $(DEBUG_FLAGS) -o $(BIN_DIR)/$@ trt_onnx.cpp -I/include/ -I./ -I$(RT_INC) -I$(OPENCV_INC) -I$(CUDA_INC) -L$(CUDA_LIB) -L$(RT_LIB) \
	-lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lglog -lnvinfer -lcudart -lnvonnxparser
