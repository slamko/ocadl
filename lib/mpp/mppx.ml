
open Ppxlib

let ext_fun ~ctxt arr =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  Ast_builder.Default.(pexp_array ~loc arr)

let extracter () = Ast_pattern.(single_expr_payload (ppat_construct __))

let exten = Extension.V3.declare
              "mat"
              Extension.Context.Expression
              (extracter ())
              ext_fun

let rule = Context_free.Rule.extension exten

let () = 
  Ppxlib.Driver.register_transformation ~rules:[rule] "mat";
  print_endline "Hello, World!";
  ()
