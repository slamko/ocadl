LIB =
LIB += OpenCL

CPP_SRC =
CPP_SRC += ocl.cpp
CPP_SRC += blac.cpp
CPP_SRC += deep.cpp

OBJS = $(patsubst %.cpp,%.o,$(CPP_SRC))

all: lib

test: $(CPP_SRC)
	g++ $^ -O2 -o test -msse4.2 -lOpenCL

lib: $(OBJS)
	ar -rc libgpu.a $(OBJS)

$(OBJS): $(CPP_SRC)
	g++ -fPIC $^ -msse4.2 -c

clean:
	$(RM) -f *.out
	$(RM) -f *.o
	$(RM) -f *.a
	$(RM) -f *.so
