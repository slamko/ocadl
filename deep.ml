open Sys
open List
open Lacaml.D
open Random
open Unix

type nnet = {
    wl : mat list;
    bl : mat list
  }

type feed_forward = {
    res : mat list;
    wl_ff : mat list;
    bl_ff : mat list
  }

let mat_print mat =
   Format.printf
    "\
      @[<2>Matrix :\n\
        @\n\
        %a@]\n\
      @\n"
    Lacaml.Io.pp_fmat mat

let nn_print nn =
  print_string "\nNN print: \n" ;
  print_string "Weights:\n" ;
  iter mat_print nn.wl ;
  print_string "\nBiases:\n" ;
  iter mat_print nn.bl 

    
let make_nn arch : nnet =

  let rec make_wl_rec arch nn_acc =
    match arch with
    | [] -> nn_acc
    | [a] -> nn_acc 
    | h::t ->
       make_wl_rec t (Mat.random (hd t) h :: nn_acc)
  in

  let rec make_bl_rec arch nn_acc =
    match arch with
    | [] -> nn_acc
    | [a] -> nn_acc 
    | h::t ->
       make_bl_rec t (Mat.random 1 h :: nn_acc)
  in

  {
    wl = make_wl_rec (rev arch) [] ;
    bl = make_bl_rec (rev arch) [] ;
  }

let sigmoid (x : float) : float =
  Float.add 1.0 (exp (Float.neg x)) |> Float.div 1.0 

let forward_layer input wmat bmat =
  gemm input wmat |> Mat.add bmat |> Mat.map sigmoid

let forward input nn =

  let rec forward_rec wl bl input acc =
    match wl with
    | [] -> acc
    | hw::tw ->
       let hb = hd bl in
       (* mat_print hw ; *)
       (* mat_print input ; *)
       let layer_activation = forward_layer input hw hb in
       forward_rec tw (tl bl) layer_activation
         { wl_ff = hw :: acc.wl_ff ;
           bl_ff = hb :: acc.bl_ff ;
           res = layer_activation :: acc.res
         }
  in

  forward_rec nn.wl nn.bl input { wl_ff = []; bl_ff = []; res = [input] }
    
let nn_of_ff ff_tree =
  { wl = ff_tree.wl_ff;
    bl = ff_tree.bl_ff;
  }

let () =
  time () |> int_of_float |> Random.init ;
  let nn = make_nn [2; 2; 1] |> forward (Mat.random 1 2) in
  mat_print (hd nn.res) ;
  ()


