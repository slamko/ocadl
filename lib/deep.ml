open List
open Types
open Nn
open Domainslib
open Deepmath
open Common
open Ppxlib

type backprop_neuron = {
    wmat_arr : float array array;
    pd_prev_arr : float array
  }

type backprop_layer = {
    prev_diff : ff_input_type;
    grad : layer_params;
  }

let fully_connected_forward meta params tens =
  let open Fully_Connected in
  Mat.mult tens params.weight_mat
  |> Mat.add params.bias_mat
  |> Mat.map meta.activation
  |> make_tens1

let pooling_forward meta tens =
  let open Mat in
  let open Pooling in

  let rec pool_rec meta mat (Row cur_row) (Col cur_col) acc =
    if cur_row >= (dim1 acc |> get_row)
    then acc
    else if cur_col >= (dim2 acc |> get_col)
    then pool_rec meta mat (Row (cur_row + 1)) (Col 0) acc
    else mat
         |> shadow_submatrix
              (Row (cur_row * meta.stride))
              (Col (cur_col * meta.stride))
              meta.filter_shape.dim1 meta.filter_shape.dim2
         |> (fun subm -> fold_left meta.fselect (get_first subm) subm)
         |> set_bind (Row cur_row) (Col cur_col) acc
         |> pool_rec meta mat (Row cur_row) (Col (cur_col + 1))  
  in

    map (fun mat ->
           zero_of_shape meta.out_shape
           |> pool_rec meta mat (Row 0) (Col 0)) tens
    |> make_tens3

let conv3d_forward meta params tens =
  let open Mat in
  let open Conv2D in
  let res_mat = map
                  (fun _ -> zero_of_shape meta.out_shape)
                  params.kernels in

  mapi
    (fun _ (Col c) mat ->
      let kernel = shadow_submatrix (Row c) (Col 0)
                     (Row 1) (dim2 params.kernels)
                     params.kernels in

          (* show_mat mat |> print_string; *)
      fold_left2 (fun acc kern in_ch ->
          let re = convolve in_ch
                     ~stride:meta.stride
                     ~padding:meta.padding
                     meta.out_shape kern
          |> add acc in
          re 
        ) mat kernel tens
      |> add_const (params.bias_mat |> get (Row 0) (Col c))
      |> map meta.act 
    ) res_mat
  |> make_tens3

let conv2d_forward meta params tens =
  Mat.make (Row 1) (Col 1) tens
  |> conv3d_forward meta params

let forward_layer : type a b c. a -> (a, b) layer -> b
  = fun input layer_type ->
  match layer_type with
  | Input3 _ -> input
  | FullyConnected (fc, fcp) ->
     (match input with
     | Tensor1 tens -> 
        tens
        |> fully_connected_forward fc fcp
     )
  | Conv2D (cn, cnp) -> 
     (match input with
     | Tensor3 tens -> 
        tens
        |> conv3d_forward cn cnp
     )
  | Flatten _ ->
     (match input with
     | Tensor3 tens ->
        tens
        |> Mat.flatten
        |> make_tens1
     )
  | Pooling pl ->
     (match input with
      | Tensor3 tens ->
         tens
         |> pooling_forward pl
     )

let forward input nn =

  let rec forward_rec : type a b c x. (a, b) ff_list -> a ->
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

  forward_rec nn.layers input BP_Nil 

let get_err : type t. t tensor -> float =
  fun tens ->
  match tens with
  | Tensor1 mat -> Mat.sum mat
  | Tensor3 mat ->
     Mat.fold_left (fun acc m -> Mat.sum m +. acc) 0. mat

let loss data nn =

  let rec loss_rec : type a b c. (a, b) nnet -> (a, b) train_data
                          -> float -> (float, string) result =
    fun nn data err ->
    match data with
    | [] -> Ok err
    | sample::data_tail ->
       let ff = forward (get_data_input sample) nn in
       let expected = get_data_out sample in
       let BP_Cons((lay, _, res), _) = ff in

       (match lay with
        | FullyConnected (_, _) ->
           (match res, expected with
            | Tensor1 res, Tensor1 exp -> 
               let diff =
                 Mat.sub res exp 
                 |> Mat.sum
               in
               loss_rec nn data_tail (err +. (diff *. diff))
           )
          | Conv2D (_, _) ->
             (match res, expected with
              | Tensor3 res, Tensor3 exp -> 
                 (* Mat.sub (Mat.of_array_size res) (Mat.of_array_size exp) *)
                 (* |> Mat.qto_array *)
                 (* |> make_tens3 *)
                 failwith "Non implemented"
             )
          | Flatten _ -> 
             (match res, expected with
              | Tensor1 res, Tensor1 exp -> 
                 let diff =
                   Mat.sub res exp 
                 |> Mat.sum
                 in
                 loss_rec nn data_tail (err +. (diff *. diff))
             )
          | _ ->
             failwith "Non implemented"
         )
  in

  let* loss = loss_rec nn data 0. in
  let avg_loss = List.length data |> float_of_int |> (/.) @@ loss in
  Ok avg_loss

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

let bp_fully_connected act act_prev meta params diff =
  let open Fully_Connected in
  match diff, act, act_prev with
     | Tensor1 diff_mat, Tensor1 act, Tensor1 act_prev ->
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

        let prev_diff = prev_diff_mat
                        |> reshape (Row 1) (Col (wmat |> dim1 |> get_row))
                        |> make_tens1
        in
          { prev_diff ;
            grad = FullyConnectedParams {
                       weight_mat = grad_wmat;
                       bias_mat = grad_bmat;
                     };
          }
     | _ -> invalid_arg "bp layer: Incompatible activation type."

let flatten_bp act_prev diff =
  match diff, act_prev with
  | Tensor1 diff_mat, Tensor1 _ ->
     { prev_diff = make_tens1 diff_mat;
       grad = FlattenParams;
     }
  | Tensor1 diff_mat, Tensor2 tens ->
     { prev_diff =
       tens
       |> Mat.get_shape
       |> Mat.reshape_of_shape diff_mat
       |> make_tens2;
      grad = FlattenParams;
    }
  | Tensor1 diff_mat, Tensor3 act_prev ->
     { prev_diff =
         diff_mat
         |> Mat.reshape3d act_prev
         |> make_tens3;
       grad = FlattenParams;
     }
  | _ -> failwith "No to tensor4 (flatten)."

let pooling_bp meta act_prev diff =
  let open Mat in
  let open Pooling in

  let diff_mat = match diff with
    | Tensor3 tens | Tensor4 tens -> tens
    | _ -> failwith "Incompatible diff mat type for pooling backprop"
  in

  let rec pool_rec meta mat diff_mat (Row cur_row) (Col cur_col) acc =
    if cur_row >= (dim1 diff_mat |> get_row)
    then acc
    else if cur_col >= (dim2 diff_mat |> get_col)
    then pool_rec meta mat diff_mat (Row (cur_row + 1)) (Col 0) acc
    else
      let cur_diff = get (Row cur_row) (Col cur_col) diff_mat in

      let res_submat = acc 
                       |> shadow_submatrix
                            (Row (cur_row * meta.stride))
                            (Col (cur_col * meta.stride))
                            meta.filter_shape.dim1 meta.filter_shape.dim2
      in

      mat
      |> shadow_submatrix
           (Row (cur_row * meta.stride))
           (Col (cur_col * meta.stride))
           meta.filter_shape.dim1 meta.filter_shape.dim2
      |> meta.fderiv meta.filter_shape cur_diff res_submat ;

      (* set (Row 0) (Col 0) res_submat 1.0 ; *)

      pool_rec meta mat diff_mat (Row cur_row) (Col (cur_col + 1)) acc
  in

  match act_prev with
  | Tensor3 act_prev | Tensor4 act_prev ->
     let prev_diff =
       map2 (fun input diff ->
           (* print diff ; *)
           pool_rec meta input diff (Row 0) (Col 0)
             (zero_of_shape (get_shape input)))
         act_prev diff_mat 
       (* |> make_tens3 *)
     in
     (* iter (fun m -> show_mat m |> Printf.printf "Diff: %s\n") prev_diff ; *)
     { prev_diff = prev_diff |> make_tens3; grad = PoolingParams}
  | _ -> failwith "Invalid previous activation for pooling bp."

let conv2d_bp meta params (prev_layer: layer option) act act_prev diff_mat =
  let open Mat in
  let open Conv2D in
  match act, act_prev, diff_mat with
  | Tensor3 act, Tensor3 act_prev, Tensor3 diff_mat ->
     let dact_f = map (map meta.deriv) act in  
     let dz = map2 hadamard dact_f diff_mat in
     let bias_mat = map sum dz in
     let kernels = map3 (fun inp out kern ->
                       convolve inp ~padding:meta.padding
                         ~stride:meta.stride (get_shape kern) out)
                     act_prev dz params.kernels
     in


     (* let prev_diff = map3 (fun inp dout kern -> *)
                         (* let kern_rot = rotate180 kern in *)
                         (* convolve dout ~padding:meta.padding *)
                           (* ~stride:meta.stride (get_shape inp) kern_rot) *)
                       (* act_prev dz params.kernels *)
                   (* |> make_tens3 *)

     let prev_diff =
       match prev_layer with
       | Some x ->
          begin match x with
          | Input _ -> diff_mat |> make_tens3 
          | _ ->
             map3 (fun inp dout kern ->
                         let kern_rot = rotate180 kern in
                         convolve dout ~padding:meta.padding
                           ~stride:meta.stride (get_shape inp) kern_rot)
                       act_prev dz params.kernels
             |> make_tens3
          end
       | None -> diff_mat |> make_tens3  
     in

     { prev_diff;
       grad = Conv2DParams
                { kernels;
                  bias_mat
                }
     }
                         
  | Tensor3 m, Tensor2 a, Tensor3 _ -> failwith "conv2d_bp: activation type"
  | _ -> failwith "conv2d_bp: invalid activation type"

let backprop_layer layer (prev_layer : layer option) act_list diff_mat =
  match layer with
  | Input _ ->
     { prev_diff = diff_mat;
       grad = InputParams;
     }
  | FullyConnected (meta, params) ->
     let act = hd act_list in
     let act_prev = hdtl act_list in
     bp_fully_connected act act_prev meta params diff_mat
  | Conv2D (meta, params) ->
     let act = hd act_list in
     let act_prev = hdtl act_list in
     conv2d_bp meta params prev_layer act act_prev diff_mat 
  | Pooling meta ->
     let act_prev = hdtl act_list in
     pooling_bp meta act_prev diff_mat
  | Flatten _ -> 
     let act_prev = tl act_list |> hd in
     flatten_bp act_prev diff_mat
 
let rec backprop_nn :
          type a b c x. ((b, a) layer * b * a, c) bp_list -> a ->
               (a, x) param_list ->
               (b, x) param_list =
  
  fun bp_list diff grad_acc ->
  match bp_list with
  | BP_Nil ->
     grad_acc
  | BP_Cons ((lay, input, out), tail) ->
     let prev_lay = if List.length bp_layers > 1
                    then Some (hdtl bp_layers)
                    else None
     in

     let bp_layer = backprop_layer lay prev_lay act_list 
                      diff in

     let param_list = bp_layer.grad::grad_acc.param_list in
     let prev_diff_mat = bp_layer.prev_diff in
     
     backprop_nn tail 
       {param_list} prev_diff_mat

  
let nn_gradient nn data  =

  let rec bp_rec : type a b c. (a, b) nnet ->
                        (a, b) train_data ->
                        (a, b) nnet_params -> (a, b) nnet_params
    = fun nn data bp_grad_acc ->
    match data with
    | [] -> bp_grad_acc
    | cur_sample::data_tail ->
       let ff_net      = forward (get_data_input cur_sample) nn in
       (* show_nnet ff_net.backprop_nn |> print_string; *)
       (* Printf.printf "len %d\n" @@ List.length ff_net.res ; *)
       let BP_Cons ((lay, _, ff_res), _)       = ff_net in
       let expected_res = get_data_out cur_sample in

       let res_diff : b =
         (match lay with
          | FullyConnected (_, _) ->
             (match ff_res, expected_res with
              | Tensor1 res, Tensor1 exp -> 
                 Mat.sub res exp 
                 |> make_tens1
             )
          | Conv2D (_, _) ->
             (match ff_res, expected_res with
              | Tensor3 res, Tensor3 exp -> 
                 (* Mat.sub (Mat.of_array_size res) (Mat.of_array_size exp) *)
                 (* |> Mat.qto_array *)
                 (* |> make_tens3 *)
                 failwith "Non implemented"
             )
          | Flatten _ -> 
             (match ff_res, expected_res with
              | Tensor1 res, Tensor1 exp -> 
                 Mat.sub res exp
                 |> make_tens1
             )
         ) in

       let bp_grad = { param_list = backprop_nn ff_net res_diff PL_Nil } in
       
       nn_params_apply Mat.add bp_grad bp_grad_acc
       |> bp_rec nn data_tail
   
  in
 
  let newn = bp_rec nn data @@ nn_zero_params nn in
  let scale_fact =
    List.length data
    |> float_of_int
    |> (fun x -> 1. /. x) in
  (* print_endline "Full nn"; *)
  let param_nn = nn_params_map (( *. ) scale_fact) newn in
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
 *)
 

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
       |> List.fold_left (nn_params_apply Mat.add) grad_acc
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
         |> nn_params_map @@ Mat.scale learning_rate
         |> nn_apply_params Mat.sub nn 
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
  (* let rec_dom_cnt = Domain.recommended_domain_count () in *)
  let rec_dom_cnt = 2 in
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
    (* match check_nn_geometry nn.data data with *)
    (* | Ok _ ->  *)
       let pool = Task.setup_pool ~num_domains: domains_num () in

       let learn_task =
         (fun _ -> learn_rec pool domains_num data
                    epoch_num learning_rate batch_size
                    0 (nn_zero_params nn) nn)
       in
       
       let res = Task.run pool learn_task in
       
       Task.teardown_pool pool ;
       Ok res
    (* | Error err -> Error err *)

(*
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

  *)  
