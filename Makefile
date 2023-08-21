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

all: $(SRC)
	gcc	-c vector/blac.c
	gcc	-c -I/home/slamko/.opam/default/lib/ocaml vector/gemm.c
	ocamlopt -I lib -I lib/layers -I lib/math blac.o gemm.o $(SRC) -cclib -lOpenCL
