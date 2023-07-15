
all: deep

deep.cmx: deep.ml
	ocamlopt -c -I ~/.opam/default/lib/lacaml $^ -o $@

deep: deep.cmx
	ocamlopt -o $@ -I ~/.opam/default/lib/lacaml unix.cmxa lacaml.cmxa $^
