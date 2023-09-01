open Common
open Deepmath
open Deep
open Nn
open Alias
open Types
open Tensor

(*
let xor_in =
  [
    data [|0.; 0.|] ;
    data [|0.; 1.|] ;
    data [|1.; 0.|] ;
    data [|1.; 1.|] ;
  ]
 *)
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
    |> make_fully_connected ~ncount:256 ~actf:Sigmoid
    |> make_fully_connected ~ncount:128 ~actf:Sigmoid
    |> make_fully_connected ~ncount:64 ~actf:Sigmoid
    (* |> make_fully_connected ~ncount:16 ~act:sigmoid ~deriv:sigmoid' *)
    (* |> make_fully_connected ~ncount:16 ~act:sigmoid ~deriv:sigmoid' *)
    (* |> make_fully_connected ~ncount:3 ~act:sigmoid ~deriv:sigmoid' *)
    |> make_fully_connected ~ncount:10 ~actf:Sigmoid
    |> make_nn in

  let conv_nn =
    make_input2d (Mat.make_shape (Row 28) (Col 28)) 

    |> make_conv2d ~padding:0 ~stride:1 ~act:Relu
         ~kernel_shape:(Mat.make_shape (Row 2) (Col 2))

    |> make_pooling2d ~stride:1 ~f:Max
         ~filter_shape:(Mat.make_shape (Row 2) (Col 2))

    |> make_pooling2d ~stride:1 ~f:Max
         ~filter_shape:(Mat.make_shape (Row 2) (Col 2))
(*

    |> make_conv2d ~padding:1 ~stride:1 ~act:Relu
         ~kernel_shape:(Mat.make_shape (Row 8) (Col 8))

    |> make_pooling2d ~stride:1 ~f:Max
         ~filter_shape:(Mat.make_shape (Row 2) (Col 2))


    |> make_conv2d ~padding:1 ~stride:1 ~act:relu ~deriv:relu'
         ~kernel_shape:(make_shape (Row 4) (Col 4))
         ~kernel_num:1

    |> make_pooling ~stride:2 ~f:pooling_max ~fbp:pooling_max_deriv
         ~filter_shape:(make_shape (Row 4) (Col 4))
 *)

    |> make_flatten2d
    |> make_fully_connected ~ncount:16 ~actf:Sigmoid
    |> make_fully_connected ~ncount:16 ~actf:Sigmoid
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

  (* nn_print trained_nn.layers ; *)

  nn_print trained_nn ;
  perform trained_nn train_data ;

  Printf.printf "initial loss: %f\n" res ;
  Printf.printf "trained loss: %f\n" new_res ;


  (* Printf.printf "Act: %d\n" (actf_to_enum Sigmoid) ; *)

  (* Printf.printf "M1: \n%!" ; *)
  (* cc_mat_print m1.matrix ; *)
  (* Printf.printf "Scaled: %d \n%!" @@ Bigarray.Array2.dim2 scaled; *)
  (* cc_mat_print scaled ; *)
  (* perform nn train_data ; *)
  (* let m1 = Mat.random (Row 64) (Col 64) in *)
  (* let m2 = Mat.random (Row 64) (Col 64) in *)
  (* let res = Mat.random (Row 64) (Col 64) in *)
  (* let r = cc_mat_mul (allocate float m1) *)

  Ok ()
 

  (* if not @@ String.equal !save_file "" *)
  (* then save_to_json !save_file trained_nn.data; *)

 (* match res with *)
  (* | Ok loss -> Printf.printf "Ok %f\n" loss *)
  (* | Error err -> Printf.eprintf "error: %s\n" err *)

  


