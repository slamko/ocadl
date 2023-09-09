SRC =
SRC += lib/common.ml
SRC += lib/math/tensor.ml
SRC += lib/alias.ml
SRC += lib/math/shape.mli
SRC += lib/math/shape.ml

SRC += $(wildcard lib/layers/*.ml)
SRC += lib/types.ml
SRC += lib/math/deepmath.ml
SRC += lib/nn.ml
SRC += lib/deep.ml
SRC += lib/test.ml
# SRC += lib/ocadl.ml

C_SRC =
C_SRC += gpu/gemm.c

C_INCL = /home/slamko/.opam/default/lib/ocaml

CPP_SRC =
CPP_SRC += gpu/ocl.cpp
CPP_SRC += gpu/blac.cpp
CPP_SRC += gpu/deep.cpp

OBJS =
OBJS += ocl.o
OBJS += blac.o
OBJS += deep.o
OBJS += gemm.o

CAML_PKGS = 
CAML_PKGS =csv,domainslib,unix,ppx_deriving.show,ppx_deriving.enum

CFLAGS = -O2 -D CL_TARGET_OPENCL_VERSION=300 -D CL_HPP_TARGET_OPENCL_VERSION=300
CAML_FLAGS = -O2

debug: FLAGS = -g

all: $(SRC)
	gcc	-c $(CFLAGS) -I$(C_INCL) $(C_SRC)
	g++ -c $(CFLAGS) -I$(C_INCL) $(CPP_SRC)

	ocamlfind ocamlopt $(CAML_FLAGS) -o ocadl \
		-I lib -I lib/layers -I lib/math -I test $(OBJS) \
		-linkpkg -package $(CAML_PKGS) \
		$(SRC) -cclib -lOpenCL -cclib -lstdc++

dune: $(SRC)
	make -C gpu
	rm -rf _build/default/lib/lib*	
	export PKG_CONFIG_PATH=/home/slamko/proj/ai/gocadl/gpu:$PKG_CONFIG_PATH && dune build

clean:
	dune clean
	rm -f lib/*.o
	rm -f lib/*.cmi
	rm -f lib/*.cma
	rm -f lib/*.cmxa
	rm -f lib/*.cmx
	rm -f lib/*.cmt
	rm -f *.o
	rm -f *.cmi
	rm -f *.cma
	rm -f *.cmxa
	rm -f *.cmx
	rm -f *.cmt

