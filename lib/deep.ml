open Lacaml.D
open List
open Types
open Nn
open Domainslib
open Deepmath

type backprop_neuron = {
    wmat_arr : float array array;
    pd_prev_arr : float array
  }

type backprop_layer = {
    prev_diff_arr : float array;
    grad : layer_grad;
  }

let forward_layer input = function
  | FullyConnected fc ->
     [
       gemm (hd input) fc.data.weight_mat
       |> Mat.add fc.data.bias_mat
       |> Mat.map fc.activation ]
  | Conv2D cn -> List.map2
                   (fun in_mat kern ->
                     convolve in_mat ~stride:cn.stride kern)
                   input cn.data.kernels
  (* | Pooling pl ->  *)

let forward (input : mat list) nn =

  let rec forward_rec layers input acc =
    match layers with
    | [] -> acc
    | lay::t ->
       let cur_layer = fst lay in
       let layer_activation = forward_layer input cur_layer in
       forward_rec t layer_activation
         { res = layer_activation :: acc.res;
           layers_ff = cur_layer::acc.layers_ff;
         }
  in

  forward_rec nn.layers input
    {res = [input];
     layers_ff = [];
    }

let cost (data : train_data) nn =

  let rec cost_rec nn data err = 
    match data with
    | [] -> err
    | sample::data_tail ->
       let ff = forward (get_data_input sample) nn in
       let expected = get_data_out sample in
       let diff = Mat.sub (hd (hd ff.res)) expected
                  |> Mat.as_vec
                  |> Vec.fold (fun res total -> res +. total) 0. in

       cost_rec nn data_tail (err +. (diff *. diff))
  in

  List.length data |> float_of_int |> (/.) @@ cost_rec nn data 0.

let backprop_neuron w_mat_arr fderiv w_row w_col ff_len diffi ai ai_prev_arr
      pd_prev_arr_acc : backprop_neuron =

  let rec bp_neuron_rec irow w_col diffi ai ai_prev_arr =
    match irow with
    | -1 -> ()
    | _ ->
       let ai_prev = Array.get ai_prev_arr irow in
       let ai_deriv = fderiv ai in
       let wi_grad : float = 2. *. diffi *. ai_deriv *. ai_prev in
      (* Printf.printf "Diff %f\nNeuron WI : %f \nrow %d \ncol %d \nai %f \nai prev %f\n" diffi wi_grad irow w_col ai ai_prev; *)
       
       let calc_pd () = 
         if ff_len > 1
         then
           let cur_w = w_mat_arr.(irow).(w_col) in
           let pd_prev : float = 2. *. diffi *. ai_deriv *. cur_w in
           let last_pd_prev = Array.get pd_prev_arr_acc irow in
           pd_prev_arr_acc.(irow) <- (pd_prev +. last_pd_prev) ;
           (* Printf.printf "calc prev %d\n" ff_len ; *)
       in

       calc_pd ();
       w_mat_arr.(irow).(w_col) <- wi_grad ;

       bp_neuron_rec (irow - 1) w_col diffi ai ai_prev_arr
  in

  bp_neuron_rec w_row w_col diffi ai ai_prev_arr ;
  {
    wmat_arr = w_mat_arr;
    pd_prev_arr = pd_prev_arr_acc
  }
  
let rec backprop_layer w_row w_col (wmat_arr : float array array)
          (fderiv : deriv)
          (b_acc_arr : float array)
          (prev_diff_arr_acc : float array)
          (ai_arr : float array)
          (diff_arr : float array)
          (ai_prev_arr : float array)
          ff_len i : backprop_layer =

  if i = w_col
  then { prev_diff_arr = prev_diff_arr_acc ;
         wmat = Mat.of_array wmat_arr;
         bmat = Mat.of_array [|b_acc_arr|]
       }
  else
    let ai = Array.get ai_arr i in
    let diff = Array.get diff_arr i in
    let bp_neuron = backprop_neuron wmat_arr fderiv (w_row - 1) i ff_len
                      diff ai ai_prev_arr prev_diff_arr_acc in
    let grad_mat = bp_neuron.wmat_arr in
    let pd_prev_diff_arr = bp_neuron.pd_prev_arr in
    let bias_grad = 2. *. diff *. ai *. (1. -. ai) in

    Array.set b_acc_arr i bias_grad ;
    (* print_endline "Diff arr"; *)
    (* arr_print diff_arr ; *)
    (* print_float @@ Array.get b_acc_arr 0; *)

    backprop_layer w_row w_col grad_mat fderiv b_acc_arr
      pd_prev_diff_arr ai_arr diff_arr ai_prev_arr ff_len (i + 1)
      

let rec backprop_nn (ff_list : mat list)
          (wmat_list : weight_list)
          (deriv_list : deriv list)
          (wgrad_mat_list_acc : weight_list)
          (bgrad_mat_list_acc : bias_list)
          diff_vec : nnet_grad =

  match ff_list with
  | [_] | [] ->
     { wl = wgrad_mat_list_acc ;
       bl = bgrad_mat_list_acc
     }
  | cur_activation::ff_tail ->
     let wmat = hd wmat_list in
     let fderiv = hd deriv_list in
     let wrows = Mat.dim1 wmat in
     let wcols = Mat.dim2 wmat in

     let bp_layer = backprop_layer wrows wcols (Mat.to_array wmat)
                      fderiv
                      (Array.make wcols 0.) (Array.make wrows 0.)
                      (mat_row_to_array 0 cur_activation) diff_vec
                      (mat_row_to_array 0 (hd ff_tail))
                      ((List.length ff_list) - 1) 0 in
     (* print_endline "After"; *)
     (* mat_print bp_layer.wmat ; *)

     let wgrad_list = bp_layer.wmat :: wgrad_mat_list_acc in
     let bgrad_list = bp_layer.bmat :: bgrad_mat_list_acc in

     backprop_nn ff_tail (tl wmat_list) (tl deriv_list)
       wgrad_list bgrad_list bp_layer.prev_diff_arr
  
let nn_gradient nn data : nnet_grad=

  let rec bp_rec nn data bp_grad_acc : nnet_grad =
    match data with
    | [] -> bp_grad_acc
    | cur_sample::data_tail ->
       let ff_net       = forward (get_data_input cur_sample) nn in
       let ff_res       = hd ff_net.res in
       let expected_res = get_data_out cur_sample in
       let res_diff     = Mat.sub ff_res expected_res |>
                            mat_row_to_array 0 in

       let bp_grad = backprop_nn ff_net.res ff_net.wl_ff
                       nn.derivatives [] [] res_diff in

       (* Printf.printf "One sample nn" ; *)
       nn_apply mat_add bp_grad bp_grad_acc |> bp_rec nn data_tail
  in
 
  let newn = nn_grad_zero  
            |> bp_rec nn data in

  (* print_endline "Full nn"; *)
  newn |> nn_grad_map (mat_scale (List.length data
             |> float_of_int
             |> (fun x -> 1. /. x)))

let check_nn_geometry nn data =
  let sample = hd data in

  let single_dim_mat_len m =
    m
    |> Mat.to_array
    |> arr_get 0
    |> Array.length
  in
  
  print_int (get_data_input sample |> single_dim_mat_len);
  if get_data_input sample |> single_dim_mat_len = Mat.dim1 (hd nn.wl)
  then
    if get_data_out sample |> single_dim_mat_len = Mat.dim2 (hd (rev nn.wl))
    then Ok nn
    else Error "Unmatched data geometry: number of output neurons should be equal to number of expected outputs"
  else Error "Unmatched data geometry: number of input neurons should be equal to number of data inputs."
 
let rec learn_rec pool pool_size data epoch_num
          learning_rate batch_size pools_per_batch grad_acc nn =
  match epoch_num with
  | 0 -> nn
  | _ ->

     let rec spawn_bp_pool i tasks =
       match i with
       | 0 -> tasks
       | _ ->
          tasks
          |> cons @@
               Task.async pool
                 (fun _ ->
                   nn_gradient nn data
                   |> nn_map @@ mat_scale learning_rate)
          |> spawn_bp_pool (i - 1)
     in

     let cycles_to_batch = batch_size - (pools_per_batch * pool_size) in
     let thread_num =
       if cycles_to_batch > pool_size
       then pool_size
       else cycles_to_batch
     in

     let cur_domain_num =
       if thread_num > epoch_num
       then epoch_num
       else thread_num
     in

     let task_list = spawn_bp_pool cur_domain_num []
     in

     let grad_list = 
       task_list
       |> List.map @@ Task.await pool in

     (* print_string "hello\n"; *)
     let full_grad =
       grad_list
       |> nn_list_fold_left mat_add
       |> nn_apply mat_add grad_acc
     in
     
     let batch_grad = if cycles_to_batch = cur_domain_num
                      then nn_zero grad_acc
                      else full_grad 
     in

     let new_nn_data = if cycles_to_batch = cur_domain_num
                  then nn_apply mat_sub nn.data full_grad
                  else nn.data
     in

     let new_nn = {
         data = new_nn_data;
         activations = nn.activations;
         derivatives = nn.derivatives
       }
     in

     (* nn_print new_nn ; *)

     let next_batch_epoch = if cycles_to_batch = cur_domain_num
                            then 0
                            else pools_per_batch + 1
     in

     learn_rec pool pool_size data (epoch_num - cur_domain_num)
       learning_rate batch_size next_batch_epoch batch_grad new_nn

let recomended_domain_num =
  let rec_dom_cnt = Domain.recommended_domain_count () in
  if rec_dom_cnt = 0
  then 1
  else rec_dom_cnt

let (>>=) a f =
  match a with
  | Ok value -> Ok (f value)
  | Error err -> Error err

let learn data ?(epoch_num = 11) ?(learning_rate = 1.0) ?(batch_size = 2)
      nn =
  
  let domains_num =
    if batch_size > recomended_domain_num
    then recomended_domain_num
    else batch_size
  in

  if batch_size > epoch_num
  then Error "Batch size greater than the number of epochs"
  else
    match check_nn_geometry nn.data data with
    | Ok _ -> 
       let pool = Task.setup_pool ~num_domains: domains_num () in

       let learn_task =
         (fun _ -> learn_rec pool domains_num data
                    epoch_num learning_rate batch_size
                    0 (nn_zero nn.data) nn)
       in
       
       let res = Task.run pool learn_task in
       
       Task.teardown_pool pool ;
       Ok res
    | Error err -> Error err
