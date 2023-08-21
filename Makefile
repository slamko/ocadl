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

all: $(SRC)
	gcc	-c vector/blac.c
	gcc	-c -I/home/slamko/.opam/default/lib/ocaml vector/gemm.c

	ocamlfind ocamlopt -o ocadl \
		-I lib -I lib/layers -I lib/math -I test \
		blac.o gemm.o \
		-linkpkg -package csv,domainslib,unix \
		$(SRC) -cclib -lOpenCL
