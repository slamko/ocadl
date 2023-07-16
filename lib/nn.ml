open Lacaml.D
open Types
open Deepmath

let nn_print nn =
  print_string "\nNN print: \n" ;
  Printf.printf "Weights:\n" ;
  List.iter mat_print nn.wl ;
  Printf.printf "\nBiases:\n" ;
  List.iter mat_print nn.bl 
    
let make_nn arch : nnet =

  let rec make_wl_rec arch nn_acc =
    match arch with
    | [] -> nn_acc
    | [_] -> nn_acc 
    | h::t ->
       make_wl_rec t (Mat.random (List.hd t) h :: nn_acc)
  in

  let rec make_bl_rec arch nn_acc =
    match arch with
    | [] -> nn_acc
    | [_] -> nn_acc 
    | h::t ->
       make_bl_rec t (Mat.random 1 h :: nn_acc)
  in

  let rev_arch = List.rev arch in
  {    
    wl = make_wl_rec rev_arch [] ;
    bl = make_bl_rec rev_arch [] ;
  }
   
let nn_apply proc nn1 nn2 =
  {
    wl = List.map2 proc nn1.wl nn2.wl;
    bl = List.map2 proc nn1.bl nn2.bl
  }

let nn_map proc nn =
  { wl = List.map proc nn.wl ;
    bl = List.map proc nn.bl ;
  }

let nn_zero nn =
  { wl = make_zero_mat_list nn.wl;
    bl = make_zero_mat_list nn.bl
  }

let get_data_input sample =
  snd sample

let get_data_out sample =
  fst sample

