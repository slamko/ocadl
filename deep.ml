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

type backprop_neuron = {
    wmat_arr : float array array;
    pd_prev_arr : float array
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

let nn_map proc nn =
  { wl = List.map proc nn.wl ;
    bl = List.map proc nn.bl ;
  }

let make_zero_mat_list mat_list =
  List.fold_right (fun mat mlist ->
      Mat.make (Mat.dim1 mat) (Mat.dim2 mat) 0. ::  mlist) [] mat_list

let nn_zero nn =
  { wl = make_zero_mat_list nn.wl;
    bl = make_zero_mat_list nn.bl
  }

let backprop_neuron w_mat_arr w_col w_row diffi ai ai_prev_arr
      pd_prev_arr_acc =

  let rec bp_neuron_rec w_col irow diffi ai ai_prev_arr =
    match irow with
    | -1 -> {
        wmat_arr = w_mat_arr;
        pd_prev_arr = pd_prev_arr_acc
      }

    | _ ->
       let ai_prev = Array.get ai_prev_arr irow in
       let wi_grad : float = 2.0 *. diffi *. ai *. (1. -. ai) *. ai_prev in
       let cur_w = Array.get (Array.get w_mat_arr w_col) irow in
       let pd_prev : float = 2.0 *. diffi *. ai *. (1. -. ai) *. cur_w in
       let last_pd_prev = Array.get pd_prev_arr_acc irow in
       
       Array.set (Array.get w_mat_arr w_col) irow wi_grad ;
       Array.set pd_prev_arr_acc irow (pd_prev +. last_pd_prev) ;

       bp_neuron_rec w_col (irow - 1) diffi ai ai_prev_arr
  in

  bp_neuron_rec w_col (w_row - 1) diffi ai ai_prev_arr
  

let rec backprop_layer w_row w_col wmat_arr b_acc_arr prev_diff_arr_acc
          ai_arr diff_arr ai_prev_arr i =
  if i = w_col
  then { prev_diff_vec = prev_diff_arr_acc ;
         wmat = Mat.of_array wmat_arr;
         bmat = Mat.from_row_vec (Vec.of_array b_acc_arr)
       }
  else
    let ai = Array.get ai_arr i in
    let diff = Array.get diff_arr i in
    let bp_neuron = backprop_neuron wmat_arr w_row w_col diff ai
                      ai_prev_arr (Array.make w_row 0.) in
    let grad_mat = bp_neuron.wmat_arr in
    let pd_prev_diff_arr = bp_neuron.pd_prev_arr in
    let bias_grad = 2. *. diff *. ai *. (1. -. ai) in

    Array.set b_acc_arr i bias_grad ;

    backprop_layer w_row w_col grad_mat b_acc_arr pd_prev_diff_arr ai_arr
      diff_arr ai_prev_arr (i + 1)
      

let rec backprop_nn ff_list wmat_list wgrad_mat_list bgrad_mat_list
          diff_vec =
  match ff_list with
  | [_] -> { wl = wgrad_mat_list ;
             bl = bgrad_mat_list
           }
  | cur_activation::ff_tail ->
     let wmat = hd wmat_list in
     let wrows = Mat.dim1 wmat in
     let wcols = Mat.dim2 wmat in
     let bp_layer = backprop_layer wrows wcols (Mat.to_array wmat)
                      (Array.make wcols 0.) (Array.make wrows 0.)
                      (Array.get (Mat.to_array cur_activation) 0) diff_vec
                      (Array.get (Mat.to_array (hd ff_tail)) 0) 0 in

     let wgrad_list_acc = bp_layer.wmat :: wgrad_mat_list in
     let bgrad_list_acc = bp_layer.bmat :: bgrad_mat_list in

     backprop_nn ff_tail (tl wmat_list) wgrad_list_acc bgrad_list_acc bp_layer.prev_diff_vec
  

let arr_get index arr =
  Array.get arr index

let mat_add mat1 mat2 =
  Mat.add mat1 mat2

let mat_sub mat1 mat2 =
  Mat.sub mat1 mat2

let backprop nn data =

  let rec bp_rec nn data bp_grad_acc =
    match data with
    | [] -> bp_grad_acc
    | cur_sample::data_tail ->
       let ff_net       = forward (snd cur_sample) nn in
       let ff_res       = hd ff_net.res in
       let expected_res = fst cur_sample in
       let res_diff     = Mat.sub ff_res expected_res |>
                            Mat.to_array |>
                            arr_get 0 in

       let bp_grad = backprop_nn ff_net.res ff_net.wl_ff [] [] res_diff in

       nn_apply mat_add bp_grad bp_grad_acc |> bp_rec nn data_tail
  in

  nn_zero nn |> bp_rec nn data 

let rec learn nn data iter =

  match iter with
  | 0 -> nn
  | _ ->
     let grad_nn = backprop nn data in
     let new_nn = nn_apply mat_sub nn grad_nn in
     
     learn new_nn data (iter - 1)

let () =
  time () |> int_of_float |> Random.init ;
  let nn = make_nn [2; 2; 1] |> forward (Mat.random 1 2) in
  mat_print (hd nn.res) ;
  ()


