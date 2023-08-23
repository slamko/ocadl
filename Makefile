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
C_SRC += gpu/blac.c
C_SRC += gpu/deep.c
C_SRC += gpu/gemm.c

C_INCL = /home/slamko/.opam/default/lib/ocaml

FLAGS = -O2

debug: FLAGS += -g

all: $(SRC)
	gcc	-c $(FLAGS) -I$(C_INCL) $(C_SRC)

	ocamlfind ocamlopt $(FLAGS) -o ocadl \
		-I lib -I lib/layers -I lib/math -I test \
		blac.o deep.o gemm.o \
		-linkpkg -package csv,domainslib,unix \
		$(SRC) -cclib -lOpenCL
