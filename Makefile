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
SRC += lib/ocadl.ml

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

FLAGS = -O2

debug: FLAGS = -g

all: $(SRC)
	gcc	-c $(FLAGS) -I$(C_INCL) $(C_SRC)
	g++ -c $(FLAGS) -I$(C_INCL) $(CPP_SRC)

	ocamlfind ocamlopt $(FLAGS) -o ocadl \
		-I lib -I lib/layers -I lib/math -I test $(OBJS) \
		-linkpkg -package $(CAML_PKGS) \
		$(SRC) -cclib -lOpenCL -cclib -lstdc++
