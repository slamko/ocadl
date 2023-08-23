open List
open Types
open Alias
open Nn
open Domainslib
open Deepmath
open Common
open Tensor

type backprop_neuron = {
    wmat_arr : float array array;
    pd_prev_arr : float array
  }

type ('inp, 'out) backprop_layer = {
    prev_diff : 'inp;
    grad : ('inp, 'out) layer_params;
  }

let forward_layer : type a b. a -> (a, b) layer -> b
  = fun input layer_type ->
  match layer_type with
  | Input3 _ -> input
  | Input2 _ -> input
  | Input1 _ -> input
  | FullyConnected (fc, fcp) ->
     (match input with
     | Tensor1 tens -> 
        (* Vec.print tens; *)
        cc_fully_connected_ff tens.matrix
          fcp.weight_mat.matrix fcp.bias_mat.matrix
        |> Vec.create |> make_tens1
     )
  | Flatten2D meta ->
     let (Tensor2 tens) = input in
     cc_mat_flatten tens.matrix
     |> Vec.create |> make_tens1

(*
  | Conv3D (cn, cnp) -> 
     (match input with
     | Tensor3 tens -> 
        tens
        |> conv3d_forward cn cnp
     )
  | Flatten _ ->
     (match input with
     | Tensor3 tens ->
        tens
        |> Vec.to_mat
        |> Mat.flatten
        |> Mat.to_vec
        |> make_tens1
     )
  | Pooling pl ->
     (match input with
      | Tensor3 tens ->
         tens
         |> pooling_forward pl
     )
 *)

let forward input nn =

  let rec forward_rec : type a b x. (a, b) ff_list -> a ->
                             ((x, a) layer * x * a, _) bp_list ->
                             ((x, b) layer * x * b, _) bp_list
    = fun layers input acc ->
    match layers with
    | FF_Nil -> acc
    | FF_Cons (lay, tail) ->
       let act = forward_layer input lay in
       let upd_acc = BP_Cons ((lay, input, act), acc) in
       forward_rec tail act upd_acc
  in

  let act = forward_layer input nn.input in
  { bp_input = (nn.input, input, act);
    bp_data = forward_rec nn.layers input BP_Nil
  }

let tens1_error (res : Vec.t) (exp : Vec.t) =
  cc_vec_sub res.matrix exp.matrix
  |> cc_vec_sum

(*
let tens3_error res exp =
   let open Mat in
   let zero_mat =
     (res
      |> Vec.get_first
      |> get_shape
      |> zero_of_shape) in

   Vec.fold_left2
     (fun acc res exp ->
       sub res exp |> add acc) zero_mat res exp
   |> sum


let get_err : type t. t tensor -> float =
  fun tens ->
  match tens with
  | Tensor1 vec -> Vec.sum vec
  | Tensor2 mat -> Mat.sum mat
  | Tensor3 mat_vec ->
     Vec.fold_left (fun acc m -> Mat.sum m +. acc) 0. mat_vec
  | Tensor4 mat_mat ->
     Mat.fold_left (fun acc m -> Mat.sum m +. acc) 0. mat_mat

 *)
let tens1_diff (res : vec) (exp : vec) =
  cc_vec_sub res.matrix exp.matrix
  |> cc_vec_scale 2.0
  |> Vec.create
  |> make_tens1

(* let tens3_diff res exp = *)
  (* Vec.map2 Mat.sub res exp *)
  (* |> make_tens3 *)

let loss data nn =

  let rec loss_rec : type a b n. (n, a, b) nnet -> (a, b) train_data
                          -> float -> (float, string) result =
    fun nn data err ->
    match data with
    | [] -> Ok err
    | sample::data_tail ->
       let ff = forward (get_data_input sample) nn in
       let expected = get_data_out sample in

       let diff : float =
           (match ff.bp_data with
            | BP_Nil ->
               let (lay, _, res) = ff.bp_input in
               (match lay with
                | FullyConnected (_, _) ->
                   (match res, expected with
                    | Tensor1 res, Tensor1 exp -> 
                       (* Vec.print res ; *)
                       tens1_error res exp
                   )
                (*
                | Conv3D (_, _) ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_error res exp
                   )
                | Pooling _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_error res exp
                   )
                | Input3 _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_error res exp
                   )
                 *)
               )
            | BP_Cons((lay, _, res), _) ->
               (match lay with
                | FullyConnected (_, _) ->
                   (match res, expected with
                    | Tensor1 res, Tensor1 exp -> 
                       (* Vec.print res ; *)
                       tens1_error res exp
                   )
                (*
                | Conv3D (_, _) ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_error res exp
                   )
                | Flatten _ -> 
                   (match res, expected with
                    | Tensor1 res, Tensor1 exp -> 
                       tens1_error res exp
                   )
                | Pooling _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_error res exp
                   )
                | Input3 _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_error res exp
                   )
                 *)
               )
           )
         in

         loss_rec nn data_tail (err +. (diff *. diff))

       
  in

  let* loss = loss_rec nn data 0. in
  (* let avg_loss = List.length data |> float_of_int |> (/.) @@ loss in *)
  Ok loss

let fully_connected_bp prev_layer meta params (Tensor1 act_prev)
      (Tensor1 act) (Tensor1 diff_mat) =
  let open Fully_connected in
  let (prev_diff, wgrad, bgrad) =
    cc_fully_connected_bp params.weight_mat.matrix act_prev.matrix
      act.matrix diff_mat.matrix
  in
  
  { prev_diff = prev_diff |> Vec.create |> make_tens1;
    grad = FullyConnectedParams { weight_mat = wgrad |> Mat.create;
                                  bias_mat = bgrad |> Vec.create;
             }
  }

let flatten_bp prev_layer meta (Tensor2 act_prev) (Tensor1 diff) =
  { prev_diff = cc_mat_flatten_bp
                  (row act_prev.shape.dim1) 
                  (col act_prev.shape.dim2) diff.matrix
                |> Mat.create |> make_tens2;
    grad = Flatten2DParams;
  }

let backprop_layer : type a b. (a, b) layer -> bool -> a -> b -> b ->
                          (a, b) backprop_layer
  = fun layer prev_layer act_prev act diff_mat ->
  match layer with
  | Input3 _ ->
     { prev_diff = diff_mat;
       grad = Input3Params;
     }
  | Input2 _ ->
     { prev_diff = diff_mat;
       grad = Input2Params;
     }
  | Input1 _ ->
     { prev_diff = diff_mat;
       grad = Input1Params;
     }
  | Flatten2D meta ->
     flatten_bp prev_layer meta act_prev diff_mat
  | FullyConnected (meta, params) ->
     fully_connected_bp prev_layer meta params act_prev act diff_mat
(*
  | Conv3D (meta, params) ->
     conv3d_bp meta params prev_layer act act_prev diff_mat 
  | Pooling meta ->
     pooling_bp meta act_prev diff_mat
  | Flatten _ -> 
     flatten_bp act_prev diff_mat
 *)
 
let rec backprop_nn :
          type a b c x. ((b, a) layer * b * a, c) bp_list -> a ->
               (a, x) param_list ->
               (b, x) param_list =
  
  fun bp_list diff grad_acc ->
  match bp_list with
  | BP_Nil ->
     grad_acc

  | BP_Cons ((lay, input, out), tail) ->
     let prev =
       (match tail with
        | BP_Nil | BP_Cons (_, BP_Nil) -> false
        | _ -> true
       )
     in
     
     let bp_layer = backprop_layer lay prev input out diff in

     let param_list = PL_Cons (bp_layer.grad, grad_acc) in
     let prev_diff_mat = bp_layer.prev_diff in
     
     backprop_nn tail prev_diff_mat param_list

  
let nn_gradient nn data =

  let rec bp_rec : type inp out n. (n, inp, out) nnet ->
                        (inp, out) train_data ->
                        (inp, out) nnet_params -> (inp, out) nnet_params

    = fun nn data bp_grad_acc ->
    match data with
    | [] -> bp_grad_acc
    | cur_sample::data_tail ->
       let ff_net      = forward (get_data_input cur_sample) nn in

       (* show_nnet ff_net.backprop_nn |> print_string; *)
       let expected = get_data_out cur_sample in

       let res_diff : out =
           (match ff_net.bp_data with
            | BP_Nil ->
               let (lay, _, res) = ff_net.bp_input in
               (match lay with
                | FullyConnected (_, _) ->
                   (match res, expected with
                    | Tensor1 res, Tensor1 exp -> 
                       tens1_diff res exp
                   )
                (*
                | Conv3D (_, _) ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_diff res exp
                   )
                | Pooling _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_diff res exp
                   )
                | Input3 _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_diff res exp
                   )
                 *)
               )
            | BP_Cons((lay, _, res), _) ->
               (match lay with
                | FullyConnected (_, _) ->
                   (match res, expected with
                    | Tensor1 res, Tensor1 exp -> 
                       tens1_diff res exp
                   )
                (*
                | Conv3D (_, _) ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_diff res exp
                   )
                | Flatten _ -> 
                   (match res, expected with
                    | Tensor1 res, Tensor1 exp -> 
                       tens1_diff res exp
                   )
                | Pooling _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_diff res exp
                   )
                | Input3 _ ->
                   (match res, expected with
                    | Tensor3 res, Tensor3 exp -> 
                       tens3_diff res exp
                   )
                 *)
               )
           )
         in

       let bp_grad =
         { param_list = backprop_nn ff_net.bp_data res_diff PL_Nil } in
       
       nn_params_apply cc_mat_add bp_grad bp_grad_acc
       |> bp_rec nn data_tail
   
  in
 
  let newn = bp_rec nn data @@ nn_zero_params nn in
  let scale_fact =
    List.length data
    |> float_of_int
    |> (fun x -> 1. /. x) in
  (* print_endline "Full nn"; *)
  let param_nn = nn_params_scale scale_fact newn in
  Ok param_nn


let check_nn_geometry : type inp out n. (n succ, inp, out) nnet ->
                             (inp, out) train_data ->
                             ((n succ, inp, out) nnet, string) result =
  fun nn data ->
  let sample = hd data in

  let data_in = get_data_input sample in
  let data_out = get_data_out sample in

  let (inp_layer_shape, inp_data_shape)
      : (inp Shape.shape * inp Shape.shape) =
    match nn.input, data_in with
    | Input3 meta, Tensor3 _ ->
       (meta.shape, Shape.get_shape data_in)
    | Input2 meta, Tensor2 _ ->
       (meta.shape, Shape.get_shape data_in)
    | Input1 meta, Tensor1 _ ->
       (meta.shape, Shape.get_shape data_in)
    | FullyConnected (meta, _), Tensor1 _ ->
       (meta.out_shape, Shape.get_shape data_in)
    | Conv3D (meta, _), Tensor3 _ ->
       (meta.out_shape, Shape.get_shape data_in)
    | Pooling meta, Tensor3 _ ->
       (meta.out_shape, Shape.get_shape data_in)
  in

  let (out_layer_shape, out_data_shape)
      : (out Shape.shape * out Shape.shape) =
    match nn.build_layers with
    | Build_Cons (lay, _) ->
       (match lay, data_out with
        | FullyConnected (m, _) , Tensor1 _ ->
           m.out_shape, Shape.get_shape data_out
        | Flatten m, Tensor1 _ ->
           m.out_shape, Shape.get_shape data_out
        | Flatten2D m, Tensor1 _ ->
           m.out_shape, Shape.get_shape data_out
        | Input3 m, Tensor3 _ ->
           m.shape, Shape.get_shape data_out
        | Input2 meta, Tensor2 _ ->
           (meta.shape, Shape.get_shape data_out)
        | Input1 meta, Tensor1 _ ->
           (meta.shape, Shape.get_shape data_out)
        | Conv3D (m, _), Tensor3 _ ->
           m.out_shape, Shape.get_shape data_out
        | Pooling m, Tensor3 _ ->
           m.out_shape, Shape.get_shape data_out
       )
  in
 
  if Shape.shape_eq inp_layer_shape inp_data_shape
  then
    if Shape.shape_eq out_layer_shape out_data_shape 
    then Ok nn
    else Error "Unmatchaed output data geometry."
  else Error "Unmatched input data geometry"

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
                   let@ grad = nn_gradient nn data in
                   grad 
                 )
          |> spawn_bp_pool (i - 1)
     in

     Printf.printf "Epoch: %d\n%!" epoch_num ;
     flush stdout ;

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
     let full_grad = List.hd grad_list
       (* grad_list *)
       (* |> List.fold_left (nn_params_apply cc_mat_add) grad_acc *)
     in
     
     let batch_grad =
       if cycles_to_batch = cur_domain_num
       then nn_params_zero grad_acc
       else full_grad 
     in

     let new_nn =
       if cycles_to_batch = cur_domain_num
       then
         full_grad
         |> nn_params_scale learning_rate
         |> nn_apply_params cc_mat_sub nn 
       else nn
     in
     (* nn_print new_nn ; *)

     let next_batch_epoch =
       if cycles_to_batch = cur_domain_num
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

let learn data ?(epoch_num = 11) ?(learning_rate = 1.0) ?(batch_size = 1)
      nn =
  
  let domains_num =
    if batch_size > recomended_domain_num
    then recomended_domain_num
    else batch_size
  in

  if batch_size > epoch_num
  then Error "Batch size greater than the number of epochs"
  else
    match check_nn_geometry nn data with
    | Ok _ ->
       let pool = Task.setup_pool ~num_domains: domains_num () in

       let learn_task =
         (fun _ -> learn_rec pool domains_num data
                    epoch_num learning_rate batch_size
                    0 (nn_zero_params nn) nn)
       in
       
       let res = Task.run pool learn_task in
       
       Task.teardown_pool pool ;
       Ok res
    | Error err -> Error err
