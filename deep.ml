open Sys
open List
open Lacaml.D
open Random
open Array
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

type backprop_layer = {
    prev_diff_vec : float array;
    wmat : mat;
    bmat : mat 
  }

let mat_print (mat : mat)  =
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
  List.iter mat_print nn.wl ;
  print_string "\nBiases:\n" ;
  List.iter mat_print nn.bl 

    
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

let nn_apply proc nn1 nn2 =
  {
    wl = List.map2 proc nn1.wl nn2.wl;
    bl = List.map2 proc nn1.bl nn2.bl
  }

let make_zero_mat_list mat_list =
  List.fold_right (fun mat mlist ->
      Mat.make (Mat.dim1 mat) (Mat.dim2 mat) 0. ::  mlist) [] mat_list

let nn_zero nn =
  { wl = make_zero_mat_list nn.wl;
    bl = make_zero_mat_list nn.bl
  }

let backprop_layer w_row w_col w_mat b_acc_vec prev_diff_vec_acc ai_vec
          diff_vec ai_prev_vec i =
  if i = w_col
  then { prev_diff_vec = prev_diff_vec_acc ;
         wmat = w_mat;
         bmat = Mat.from_row_vec (Vec.of_array b_acc_vec)
       }
  else { prev_diff_vec = prev_diff_vec_acc ;
         wmat = w_mat;
         bmat = Mat.from_row_vec (Vec.of_array b_acc_vec)
       }
      

let rec backprop_nn ff_list wmat_list wgrad_mat_list bgrad_mat_list
          diff_vec =
  match ff_list with
  | [_] -> { wl = wgrad_mat_list ;
             bl = bgrad_mat_list
           }
  | cur_activation::ff_tail ->
     let wmat = hd wmat_list in
     let wrows = Mat.dim1 wmat in
     let bp_layer = backprop_layer in

     let wgrad_list_acc = bp_layer.wmat :: wgrad_mat_list in
     let bgrad_list_acc = bp_layer.bmat :: bgrad_mat_list in

     backprop_nn ff_tail (tl wmat_list) wgrad_list_acc bgrad_list_acc bp_layer.prev_diff_vec
  
let backprop nn data =

  let rec bp_rec nn data bp_grad_acc =
    match data with
    | [] -> bp_grad_acc
    | cur_sample::data_tail ->
       let ff_net       = forward cur_sample nn in
       let ff_res       = hd fwd_tree.res in
       let expected_res = fst cur_sample in
       let res_diff     = Mat.sub ff_res expected_res in
       let bp_grad      = backprop_nn ff_net.res ff_net.wl [] [] res_diff in

       nn_apply Mat.add bp_grad bp_rec |> bp_rec nn data_tail
  in

  nn_zero nn |> bp_rec nn data 

let () =
  time () |> int_of_float |> Random.init ;
  let nn = make_nn [2; 2; 1] |> forward (Mat.random 1 2) in
  mat_print (hd nn.res) ;
  ()


