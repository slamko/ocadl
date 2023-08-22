LIB =
LIB += OpenCL

all: main.c
	gcc $^ -g -l$(LIB) -msse4.2 -o vec 

bench: bench.c
	gcc $^ -l $(LIB) -o bench

lib: blac.c
	gcc -shared -fPIC blac.c -fPIC -l$(LIB) -o libblac.so

clean:
	$(RM) -f *.out
	$(RM) -f *.o
	$(RM) -f *.a
	$(RM) -f *.so
