CXX := clang++
TORCH := $(HOME)/libtorch
OS := $(shell uname)
CPPFLAGS := -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include 
CXXFLAGS := -std=c++11 -std=gnu++11 -pedantic -Wall -Wfatal-errors -fPIC -O3
LDFLAGS := -shared -L$(TORCH)/lib
LDLIBS := -l torch -Wl,-rpath $(TORCH)/lib

ifeq ($(OS),Linux)
 CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif
ifeq ($(OS),Darwin)
 LDFLAGS := -undefined dynamic_lookup $(LDFLAGS)
endif

lib := ktorch.so
src := ktorch.cpp ktensor.cpp kmath.cpp knn.cpp kloss.cpp kopt.cpp kmodel.cpp ktest.cpp

all: $(lib)
*.o: k.h ktorch.h
kloss.o: kloss.h
knn.o kmodel.o ktest.o: knn.h

$(lib): $(subst .cpp,.o,$(src))
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) *.o $(lib)
