open Common
open Deepmath
open Deep
open Matrix
open Nn
open Types

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

let rec perform nn data =
  match data with
  | [] -> ()
  | sample::t ->
     (* let ff = forward (get_data_input sample) nn in *)
     (* let res = ff.res |> List.hd in *)
     (* let expected = get_data_out sample in *)
     Printf.printf "NN result: \n" ;

     Printf.printf "Expected result: \n" ;
     
     perform nn t

let test train_data_fname save_file epochs learning_rate batch_size =
  let train_data =
    if Sys.file_exists train_data_fname
    then read_mnist_train_data train_data_fname
           {dim1 = (Row 28) ; dim2 = (Col 28); dim3 = 1}
    else failwith "No train file"
  in

  let base_nn =
    make_input3d @@ make_shape3d (Row 28) (Col 28) 1
    |> make_flatten
    |> make_fully_connected ~ncount:16 ~act:sigmoid ~deriv:sigmoid'
    |> make_fully_connected ~ncount:16 ~act:sigmoid ~deriv:sigmoid'
    |> make_fully_connected ~ncount:10 ~act:sigmoid ~deriv:sigmoid'
    |> make_nn in

  let conv_nn =
    make_input3d @@ make_shape3d (Row 28) (Col 28) 1
    |> make_conv2d ~padding:1 ~stride:1 ~act:relu ~deriv:relu'
         ~kernel_shape:(make_shape (Row 4) (Col 4))
         ~kernel_num:1

    |> make_pooling ~stride:2 ~f:pooling_max ~fbp:pooling_max_deriv
         ~filter_shape:(make_shape (Row 4) (Col 4))

(*
    |> make_conv2d ~padding:1 ~stride:1 ~act:relu ~deriv:relu'
         ~kernel_shape:(make_shape (Row 4) (Col 4))
         ~kernel_num:1

    |> make_pooling ~stride:2 ~f:pooling_max ~fbp:pooling_max_deriv
         ~filter_shape:(make_shape (Row 4) (Col 4))
 *)

    |> make_flatten
    |> make_fully_connected ~ncount:16 ~act:sigmoid ~deriv:sigmoid'
    |> make_fully_connected ~ncount:10 ~act:sigmoid ~deriv:sigmoid'
    |> make_nn
  in
  
  let nn =
    (* if Sys.file_exists !save_file *)
    (* then restore_nn_from_json !save_file base_nn *)
    (* else *)
      conv_nn
  in
(*  
  let trained_nn =
    match learn train_data
            ~epoch_num:epochs
            ~learning_rate:!learning_rate
            ~batch_size:!batch_size nn with
    | Ok new_nn -> new_nn
    | Error err -> failwith err
  in
 *)

  let* res = loss train_data nn in
  (* Printf.printf "Cost: %f\n" res; *)

  let* trained_nn = learn train_data
                      ~epoch_num:epochs ~learning_rate
                      ~batch_size nn in
  let* new_res = loss train_data trained_nn in

  Printf.printf "initial loss: %f\n" res ;
  Printf.printf "trained loss: %f\n" new_res ;

  Ok ()
 

  (* if not @@ String.equal !save_file "" *)
  (* then save_to_json !save_file trained_nn.data; *)

 (* match res with *)
  (* | Ok loss -> Printf.printf "Ok %f\n" loss *)
  (* | Error err -> Printf.eprintf "error: %s\n" err *)

  


