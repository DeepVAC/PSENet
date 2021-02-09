CXXFLAGS = -I /opt/conda/lib/python3.8/site-packages/torch/include -I /opt/conda/include/python3.8/ -std=c++11 -O3 

DEPS = lanms.h
CXX_SOURCES = adaptor.cpp clipper.cpp
OPENCV = `pkg-config --cflags --libs opencv4`

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)
