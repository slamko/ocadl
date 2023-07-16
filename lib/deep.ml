open List
open Lacaml.D
open Types
open Nn
open Deepmath

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
    prev_diff_arr : float array;
    wmat : mat;
    bmat : mat 
  }
 
let nn_of_ff ff_tree =
  { wl = ff_tree.wl_ff;
    bl = ff_tree.bl_ff;
  }


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

let rec perform nn data =
  match data with
  | [] -> ()
  | sample::t ->
     let ff = forward (get_data_input sample) nn in
     let res = ff.res |> hd in
     let expected = get_data_out sample in
     Printf.printf "NN result: \n" ;
     mat_print res ;

     Printf.printf "Expected result: \n" ;
     mat_print expected ;

     perform nn t

let cost (data : train_data) nn =

  let rec cost_rec nn data err = 
    match data with
    | [] -> err
    | sample::data_tail ->
       let ff = forward (snd sample) nn in
       let expected = fst sample in
       let diff = Mat.sub (hd ff.res) expected
                  |> Mat.as_vec
                  |> Vec.fold (fun res total -> res +. total) 0. in

       cost_rec nn data_tail (err +. (diff *. diff))
  in

  List.length data |> float_of_int |> (/.) @@ cost_rec nn data 0.

let backprop_neuron w_mat_arr w_row w_col diffi ai ai_prev_arr
      pd_prev_arr_acc : backprop_neuron =

  let rec bp_neuron_rec irow w_col diffi ai ai_prev_arr =
    match irow with
    | -1 -> ()
    | _ ->
       let ai_prev = Array.get ai_prev_arr irow in
       let wi_grad : float = 2. *. diffi *. ai *. (1. -. ai) *. ai_prev in
       let cur_w = Array.get (Array.get w_mat_arr irow) w_col in
       let pd_prev : float = 2. *. diffi *. ai *. (1. -. ai) *. cur_w in
       let last_pd_prev = Array.get pd_prev_arr_acc irow in
       (* Printf.printf "Diff %f\nNeuron WI : %f \nrow %d \ncol %d \nai %f \nai prev %f\n" diffi wi_grad irow w_col ai ai_prev; *)
       
       Array.set (Array.get w_mat_arr irow) w_col wi_grad ;
       Array.set pd_prev_arr_acc irow (pd_prev +. last_pd_prev) ;

       bp_neuron_rec (irow - 1) w_col diffi ai ai_prev_arr
  in

  bp_neuron_rec w_row w_col diffi ai ai_prev_arr ;
  {
    wmat_arr = w_mat_arr;
    pd_prev_arr = pd_prev_arr_acc
  }
  

let rec backprop_layer w_row w_col (wmat_arr : float array array)
          (b_acc_arr : float array)
          (prev_diff_arr_acc : float array)
          (ai_arr : float array)
          (diff_arr : float array)
          (ai_prev_arr : float array) i : backprop_layer =

  if i = w_col
  then { prev_diff_arr = prev_diff_arr_acc ;
         wmat = Mat.of_array wmat_arr;
         bmat = Mat.of_array [|b_acc_arr|]
       }
  else
    let ai = Array.get ai_arr i in
    let diff = Array.get diff_arr i in
    let bp_neuron = backprop_neuron wmat_arr (w_row - 1) i diff ai
                      ai_prev_arr prev_diff_arr_acc in
    let grad_mat = bp_neuron.wmat_arr in
    let pd_prev_diff_arr = bp_neuron.pd_prev_arr in
    let bias_grad = 2. *. diff *. ai *. (1. -. ai) in

    Array.set b_acc_arr i bias_grad ;
    (* print_endline "Diff arr"; *)
    (* arr_print diff_arr ; *)
    (* print_float @@ Array.get b_acc_arr 0; *)

    backprop_layer w_row w_col grad_mat b_acc_arr
      pd_prev_diff_arr ai_arr diff_arr ai_prev_arr (i + 1)
      

let rec backprop_nn (ff_list : mat list)
          (wmat_list : weight_list)
          (wgrad_mat_list_acc : weight_list)
          (bgrad_mat_list_acc : bias_list)
          diff_vec : nnet =

  match ff_list with
  | [] -> { wl = wgrad_mat_list_acc ;
             bl = bgrad_mat_list_acc
           }
  | [_] -> { wl = wgrad_mat_list_acc ;
             bl = bgrad_mat_list_acc
           }
  | cur_activation::ff_tail ->
     let wmat = hd wmat_list in
     let wrows = Mat.dim1 wmat in
     let wcols = Mat.dim2 wmat in

     let bp_layer = backprop_layer wrows wcols (Mat.to_array wmat)
                      (Array.make wcols 0.) (Array.make wrows 0.)
                      (mat_row_to_array 0 cur_activation) diff_vec
                      (mat_row_to_array 0 (hd ff_tail)) 0 in
     (* print_endline "After"; *)
     (* mat_print bp_layer.wmat ; *)

     let wgrad_list = bp_layer.wmat :: wgrad_mat_list_acc in
     let bgrad_list = bp_layer.bmat :: bgrad_mat_list_acc in

     backprop_nn ff_tail (tl wmat_list)
       wgrad_list bgrad_list bp_layer.prev_diff_arr
  
let backprop nn data : nnet =

  let rec bp_rec nn data bp_grad_acc =
    match data with
    | [] -> bp_grad_acc
    | cur_sample::data_tail ->
       let ff_net       = forward (get_data_input cur_sample) nn in
       let ff_res       = hd ff_net.res in
       let expected_res = get_data_out cur_sample in
       let res_diff     = Mat.sub ff_res expected_res |>
                            mat_row_to_array 0 in

       let bp_grad = backprop_nn ff_net.res ff_net.wl_ff [] [] res_diff in

       (* Printf.printf "One sample nn" ; *)
       (* nn_print bp_grad_acc ; *)
       nn_apply mat_add bp_grad bp_grad_acc |> bp_rec nn data_tail
  in
 
  let newn = nn_zero nn
            |> bp_rec nn data in

  (* print_endline "Full nn"; *)
  (* nn_print newn ; *)
  newn |> nn_map (mat_scale (List.length data
             |> float_of_int
             |> (fun x -> 1. /. x)))

let rec learn data iter nn =

  match iter with
  | 0 -> nn
  | _ ->
     let grad_nn = backprop nn data in
     let new_nn = nn_apply mat_sub nn grad_nn in
     
     (* Printf.printf "Grad_nn\n"; *)
     (* nn_print new_nn ; *)
     learn data (iter - 1) new_nn 
