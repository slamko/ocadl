LIB =
LIB += OpenCL

CPP_SRC =
CPP_SRC += ocl.cpp
CPP_SRC += blac.cpp
CPP_SRC += deep.cpp
CPP_SRC += test.cpp

test: $(CPP_SRC)
	g++ $^ -O2 -o test -msse4.2 -lOpenCL

clean:
	$(RM) -f *.out
	$(RM) -f *.o
	$(RM) -f *.a
	$(RM) -f *.so
