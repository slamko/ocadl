CPP_SRC =
CPP_SRC += ocl.cpp
CPP_SRC += blac.cpp
CPP_SRC += deep.cpp

TEST_SRC =
TEST_SRC += test.cpp

HEADERS =
HEADERS += $(wildcard *.h)
HEADERS += $(wildcard *.hpp)

OBJS := $(patsubst %.cpp,%.o,$(CPP_SRC))
TEST_OBJS := $(patsubst %.cpp,%.o,$(TEST_SRC))
CPP_SRC += test.cpp

CPP_FLAGS = -g -D CL_HPP_TARGET_OPENCL_VERSION=300
CPP_LIBS = -lOpenCL
TEST_LIBS =

all: lib

test: $(OBJS) $(TEST_OBJS) $(HEADERS)
	g++ $(OBJS) $(TEST_OBJS) -o test $(CPP_LIBS) $(TEST_LIBS)

lib: $(OBJS)
	ar -rc libgpu.a $(OBJS)

%.o: %.cpp $(HEADERS)
	g++ -fPIC $< $(CPP_FLAGS) -c

clean:
	$(RM) -f *.out
	$(RM) -f *.o
	$(RM) -f *.a
	$(RM) -f *.so
	$(RM) -f tests/*.out
	$(RM) -f tests/*.o
	$(RM) -f tests/*.a
	$(RM) -f tests/*.so

