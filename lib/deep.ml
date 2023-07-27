open List
open Types
open Nn
open Domainslib
open Deepmath
open Ppxlib

type backprop_neuron = {
    wmat_arr : float array array;
    pd_prev_arr : float array
  }

type backprop_layer =
  | End
  | LayerGrad of {
      prev_diff : mat;
      grad : layer_params;
    }

let fully_connected_forward meta (params : fully_connected_params) tens =
    Mat.mult tens params.weight_mat
    >>= Mat.add params.bias_mat
    >>| Mat.map meta.activation
    >>| make_tens1

let pooling_forward meta tens =
  let open Mat in

  let rec pool_rec meta mat (Row cur_row) (Col cur_col) acc =
    if cur_row >= (dim1 acc |> get_row)
    then Ok acc
    else if cur_col >= (dim2 acc |> get_col)
    then pool_rec meta mat (Row (cur_row + 1)) (Col 0) acc
    else mat
         |> Mat.shadow_submatrix
              (Row ((cur_row * get_row meta.filter_rows)
                    + cur_row * meta.stride))

              (Col ((cur_col * get_col meta.filter_cols)
                    + cur_col * meta.stride))

              meta.filter_rows meta.filter_cols
         >>| Mat.fold_left meta.fselect 0.
         >>| set_bind (Row cur_row) (Col cur_col) acc
         >>= pool_rec meta mat (Row cur_row) (Col (cur_col + 1))  
  in

  let mat4 =
    try
      Ok
        (Tensor4
        (tens
         |>
           Mat.map
             (fun mat ->
               let pool =
                 make
                   (Row ((dim1 mat |> get_row) / get_row meta.filter_rows))
                   (Col ((dim2 mat |> get_col) / get_col meta.filter_cols)) 0.
                 |> pool_rec meta mat (Row 0) (Col 0) in
               match pool with
               | Ok pool_res -> pool_res
               | Error err -> failwith err)))
    with Failure err -> Error err in

  mat4

let conv3d_forward (meta : conv2d_meta) params tens =
  let open Mat in

  let res =
    try
      Ok (mapi
          (fun _ (Col c) mat ->
          begin
            match Mat.convolve mat ~stride:meta.stride params.kernels.(c) with
            | Ok res -> res
            | Error err -> failwith err end) tens
          |> make_tens4)
    with
    | Failure err -> Error err
  in

  res

let conv2d_forward (meta : conv2d_meta) params tens =
  let open Mat in
  let res_mat = create
                  (Row (size tens))
                  (Col (params.kernels |> Array.length))
                  (fun r _ -> get (Row 0) (Col (get_row r)) tens) in

  conv3d_forward meta params res_mat
  
let forward_layer (input : ff_input_type) layer_type =
  match layer_type with
  | Input -> Ok input
  | FullyConnected (fc, fcp) ->
     begin match input with
     | Tensor1 tens -> tens
                       |> fully_connected_forward fc fcp
     | Tensor2 tens -> tens
                       |> Mat.flatten2d
                       |> fully_connected_forward fc fcp
     | Tensor3 tens -> Mat.flatten tens
                       |> fully_connected_forward fc fcp
     | _
       -> Error "Invalid input for fully connected layer" end
  | Conv2D (cn, cnp) -> 
     begin match input with
     | Tensor3 tens -> tens
                       |> conv2d_forward cn cnp
     | Tensor4 tens -> tens
                       |> conv3d_forward cn cnp 
     | _
       -> Error "Invalid input for convolutional layer." end
  | Pooling pl ->
     begin match input with
     | Tensor3 tens | Tensor4 tens -> pooling_forward pl tens
     | _
       -> Error "Invalid input for pooling layer."
     end

let forward input nn =

  let rec forward_rec layers input acc =
    match layers with
    | [] -> Ok acc
    | layer::tail ->

       let* act = forward_layer input layer.layer in
       let upd_acc =
         { res = act::acc.res;
           backprop_nn = build_nn (layer::acc.backprop_nn.layers)
         } in
       forward_rec tail act upd_acc
  in

  forward_rec nn.layers input
    {
      res = [input];
      backprop_nn = build_nn [];
    }

let loss (data : train_data) nn =

  let rec loss_rec nn data err = 
    match data with
    | [] -> Ok err
    | sample::data_tail ->
       let* ff = forward (get_data_input sample) nn in
       let expected = get_data_out sample in
       let res = hd ff.res in
       match res,expected with
       | Tensor2 m, Tensor2 exp | Tensor1 m, Tensor1 exp-> 
          let* diff = Mat.sub m exp
                     >>| Mat.sum in
          loss_rec nn data_tail (err +. (diff *. diff))
       | _ -> Error "Invalid output shape"
          
  in

  let* loss = loss_rec nn data 0. in
  Ok (List.length data |> float_of_int |> (/.) @@ loss)

(*
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

 *)

let bp_fully_connected act act_prev meta params diff_mat =
  match act, act_prev with
     | Tensor1 act, Tensor1 act_prev ->
        let open Mat in

        let wmat = params.weight_mat in
        let prev_diff_mat = make (dim1 wmat) (Col 1) 0. in

        let grad_wmat =
          mapi
            (fun r c weight ->
              let ai = get (Row 0) c act in
              let ai_prev = get (Row 0) (Col (get_row r)) act_prev in
              let diff  = get (Row 0) c diff_mat in
              let ai_deriv = meta.activation ai in

              let dw = 2. *. diff *. ai_deriv *. ai_prev in
              let dprev = 2. *. diff *. ai_deriv *. weight in

              get r (Col 0) prev_diff_mat
              |> (+.) dprev
              |> set r (Col 0) prev_diff_mat;
              dw
            ) wmat in

        let grad_bmat =
          mapi
            (fun r c _ ->
              let ai = get r c act in
              let diff  = get r c diff_mat in
              let db = 2. *. diff *. ai *. (1. -. ai) in
              db
            ) params.bias_mat in

        let* prev_diff = prev_diff_mat
                      |> reshape (Row 1) (Col (wmat |> dim1 |> get_row)) in
        Ok (LayerGrad
              { prev_diff ;
                grad = FullyConnectedParams {
                    weight_mat = grad_wmat;
                    bias_mat = grad_bmat;
                  };
          })
     | _ -> Error "bp layer: Incompatible activation type."

let backprop_layer layer act act_prev diff_mat :
          (backprop_layer, string) result  =
  match layer with
  | Input -> Ok End
  | FullyConnected (meta, params) ->
     bp_fully_connected act act_prev meta params diff_mat
  | Conv2D (meta, params) -> Ok End
  | Pooling _ -> Ok End 

let rec backprop_nn (ff_list : ff_input_type list)
          (bp_layers : layer_common list) (grad_acc : nnet_params)
          (diff_mat : mat) =

  match ff_list with
  | [_] | [] ->
     Ok grad_acc
  | cur_activation::ff_tail ->
     let lay = hd bp_layers in
     let prev_act = hd ff_tail in
     let* bp_layer = backprop_layer lay.layer cur_activation
                      prev_act diff_mat in

     match bp_layer with
     | LayerGrad bp_layer ->
        let param_list = bp_layer.grad::grad_acc.param_list in
        let prev_diff_mat = bp_layer.prev_diff in
        
        backprop_nn ff_tail (tl bp_layers)
          {param_list} prev_diff_mat

     | End -> Ok grad_acc
  
let nn_gradient (nn : nnet) data  =

  let rec bp_rec nn data bp_grad_acc =
    match data with
    | [] -> Ok bp_grad_acc
    | cur_sample::data_tail ->
       let* ff_net      = forward (get_data_input cur_sample) nn in
       (* show_nnet ff_net.backprop_nn |> print_string; *)
       let ff_res       = hd ff_net.res in
       let expected_res = get_data_out cur_sample in
       match expected_res, ff_res with
       | (Tensor2 exp, Tensor2 res) | (Tensor1 exp, Tensor1 res) ->
          let* res_diff = Mat.sub res exp in

          let* bp_grad = backprop_nn ff_net.res ff_net.backprop_nn.layers
                          {param_list = []; } res_diff in

          (* Printf.printf "One sample nn" ; *)
          nn_params_apply Mat.add bp_grad bp_grad_acc
          >>= bp_rec nn data_tail
       | _ -> Error "Incompatible output shape."
  in
 
  let* newn = nn_zero_params nn
             |> bp_rec nn data in

  (* print_endline "Full nn"; *)
  let param_nn = nn_params_map
                   (fun mat ->
                     Ok (Mat.scale
                           (List.length data
                            |> float_of_int
                            |> (fun x -> 1. /. x))
                           mat)) newn in
  Ok param_nn

(*
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
    then Ok nn
    if get_data_out sample |> single_dim_mat_len = Mat.dim2 (hd (rev nn.wl))
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
 *)

let rec lern data nn epochs =
  match epochs with
  | 0 -> Ok nn
  | _ ->
     let* grad = nn_gradient nn data in
     (* show_nnet_params grad |> Printf.printf "\n\n\n\n%s\n"; *)
     (* show_nnet nn |> Printf.printf "\n\n\n\n%s\n"; *)
     let rev_grad = {param_list = grad.param_list } in
     let* new_nn = nn_apply_params Mat.sub nn rev_grad in
     lern data new_nn (epochs - 1)

let xor_in =
  [
    data [|0.; 0.|] ;
    data [|0.; 1.|] ;
    data [|1.; 0.|] ;
    data [|1.; 1.|] ;
  ]

let xor_data =
  [
    (Tensor1 (one_data 0.), Tensor1 (data [|0.; 0.|])) ;
    (Tensor1 (one_data 1.), Tensor1 (data [|0.; 1.|])) ;
    (Tensor1 (one_data 1.), Tensor1 (data [|1.; 0.|])) ;
    (Tensor1 (one_data 0.), Tensor1 (data [|1.; 1.|]))
  ]

let test () =
  Unix.time () |> int_of_float |> Random.init;

  let nn =
    make_input 1 2
    |> make_fully_connected ~ncount:3 ~act:sigmoid ~deriv:sigmoid'
    |> make_fully_connected ~ncount:1 ~act:sigmoid ~deriv:sigmoid'
    |> make_nn in

  (* show_nnet nn |> Printf.printf "nn: %s\n"; *)

  let* res = loss xor_data nn in

  let* tr_nn = lern xor_data nn 100000 in
  let* new_res = loss xor_data tr_nn in

  Printf.printf "initial loss: %f\n" res ;
  Printf.printf "trained loss: %f\n" new_res ;
  Ok ()
  (* match res with *)
  (* | Ok loss -> Printf.printf "Ok %f\n" loss *)
  (* | Error err -> Printf.eprintf "error: %s\n" err *)

  
