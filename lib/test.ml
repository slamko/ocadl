open Common
open Deepmath
open Deep
open Nn
open Alias
open Types
open Tensor
open Ctypes
open C.Functions

let xor_data =
  [
    (Tensor1 (one_data 0.), Tensor1 (data [|0.; 0.|])) ;
    (Tensor1 (one_data 1.), Tensor1 (data [|0.; 1.|])) ;
    (Tensor1 (one_data 1.), Tensor1 (data [|1.; 0.|])) ;
    (Tensor1 (one_data 0.), Tensor1 (data [|1.; 1.|]))
  ]

let rec perform : type inp out n. (n, inp, out) nnet ->
                       (inp, out) train_data -> unit =
  fun nn data ->
  match data with
  | [] -> ()
  | sample::t ->
     let ff = forward (get_data_input sample) nn in
     (* let res = ff.res |> List.hd in *)
     let expected = get_data_out sample in
     let print () =
       (match ff.bp_data with
        | BP_Nil ->
           let (lay, _, res) = ff.bp_input in
           (match lay with
            | FullyConnected (_, _) ->
               (match res, expected with
                | Tensor1 res, Tensor1 exp -> 
                   Vec.print res ;
                   Vec.print exp ;
               )
            | _ -> failwith "perf"
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
                   Vec.print res ;
                   Vec.print exp
               )
            | _ -> failwith "perf"
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
       ) in
     print ();

     perform nn t

let test train_data_fname save_file epochs learning_rate batch_size =
  gpu_init () ;
  seal matrix ;
  
  let train_data =
    if Sys.file_exists train_data_fname
    then read_mnist_train_data train_data_fname
           @@ Mat.make_shape (Row 28) (Col 28)
    else failwith "No train file"
  in

  (* let train_data = xor_data in *)

  let base_nn =
    make_input2d (Mat.make_shape (Row 28) (Col 28))
    (* make_input1d (Vec.make_shape (Col 2)) *)
    |> make_flatten2d
    |> make_fully_connected ~ncount:256 ~actf:Relu
    |> make_fully_connected ~ncount:128 ~actf:Relu
    |> make_fully_connected ~ncount:64 ~actf:Sigmoid
    (* |> make_fully_connected ~ncount:16 ~act:sigmoid ~deriv:sigmoid' *)
    (* |> make_fully_connected ~ncount:16 ~act:sigmoid ~deriv:sigmoid' *)
    (* |> make_fully_connected ~ncount:3 ~act:sigmoid ~deriv:sigmoid' *)
    |> make_fully_connected ~ncount:10 ~actf:Sigmoid
    |> make_nn in

  let conv_nn =
    make_input2d (Mat.make_shape (Row 28) (Col 28)) 

    |> make_conv2d ~padding:1 ~stride:1 ~act:Relu
         ~kernel_shape:(Mat.make_shape (Row 2) (Col 2))

    |> make_pooling2d ~stride:1 ~f:Max
         ~filter_shape:(Mat.make_shape (Row 2) (Col 2))

    |> make_conv2d ~padding:1 ~stride:1 ~act:Relu
         ~kernel_shape:(Mat.make_shape (Row 2) (Col 2))

    |> make_pooling2d ~stride:1 ~f:Avarage
         ~filter_shape:(Mat.make_shape (Row 2) (Col 2))

    |> make_flatten2d
    (* |> make_fully_connected ~ncount:16 ~actf:Sigmoid *)
    (* |> make_fully_connected ~ncount:16 ~actf:Sigmoid *)
    |> make_fully_connected ~ncount:10 ~actf:Sigmoid
    |> make_nn
  in
  
  let nn =
    (* if Sys.file_exists !save_file *)
    (* then restore_nn_from_json !save_file base_nn *)
    (* else *)
      conv_nn
  in

  let* res = loss train_data nn in
  Printf.printf "Cost: %f\n" res;

  let* trained_nn = learn train_data
                      ~epoch_num:epochs ~learning_rate
                      ~batch_size nn in
  let* new_res = loss train_data trained_nn in

  (* nn_print trained_nn; *)

  (* nn_print trained_nn ; *)
  perform trained_nn train_data ;

  Printf.printf "initial loss: %f\n" res ;
  Printf.printf "trained loss: %f\n" new_res ;

  Ok ()
 
