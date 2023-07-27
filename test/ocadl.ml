open Unix
open Ocadl__Deepmath

(*
open Ocadl__Types
open Ocadl__Nn
open Ocadl__Deepmath

let ident_data =
  [
    (data [|0.|] , data [|0.|]) ;
    (data [|1.|] , data [|1.|]) ;
    (data [|0.|] , data [|0.|]) ;
    (data [|1.|] , data [|1.|])
  ]

let xor_data =
  [
    (one_data 0., data [|0.; 0.|]) ;
    (one_data 1., data [|0.; 1.|]) ;
    (one_data 1., data [|1.; 0.|]) ;
    (one_data 0., data [|1.; 1.|])
  ]

let rec perform nn data =
  match data with
  | [] -> ()
  | sample::t ->
     let ff = forward (get_data_input sample) nn in
     let res = ff.res |> List.hd in
     let expected = get_data_out sample in
     Printf.printf "NN result: \n" ;
     mat_print res ;

     Printf.printf "Expected result: \n" ;
     mat_print expected ;

     get_data_input sample |> mat_print ;
     
     perform nn t


let usage_msg = "ocadl -l <train_data_file> -s <save_file> -i <epoch_num>
-b <batch_size> [<arch>] ... "

let epoch_num = ref 11
let batch_size = ref 1
let learning_rate = ref 0.01
let train_data_file = ref ""
let save_file = ref ""
let arch = ref []

let anon_fun layer =
  arch := (int_of_string layer)::!arch

let speclist =
  [
    ("-i", Arg.Set_int epoch_num, "Epochs count") ;
    ("-b", Arg.Set_int batch_size, "Batch size") ;
    ("-r", Arg.Set_float learning_rate, "Learning rate") ;
    ("-l", Arg.Set_string train_data_file, "Training data file name") ;
    ("-s", Arg.Set_string save_file, "Json file to dump the NN state")
  ]

let train train_data_fname epochs =
  let train_data =
    (* if Sys.file_exists train_data_fname *)
    (* then read_train_data train_data_fname 1 28 *)
    (* else *)
         xor_data
  in
  (* List.hd train_data |>  |> Mat.dim2 |> print_int ; *)
  (* let train_data = adder_data in *)
  let base_nn =
    make_input 2
    |> make_fully_connected ~ncount:2 ~act:sigmoid ~deriv:sigmoid'
    |> make_fully_connected ~ncount:1 ~act:sigmoid ~deriv:sigmoid'
    |> make_nn
  in
    
  
  let nn =
    (* if Sys.file_exists !save_file *)
    (* then restore_nn_from_json !save_file base_nn *)
    (* else *)
      base_nn
  in
  
  let trained_nn =
    match learn train_data
            ~epoch_num:epochs
            ~learning_rate:!learning_rate
            ~batch_size:!batch_size nn with
    | Ok new_nn -> new_nn
    | Error err -> failwith err
  in

  (* perform trained_nn train_data ; *)
  cost train_data nn |> Printf.printf "Cost: %f\n" ;
  trained_nn |> cost train_data |> Printf.printf "Trained Cost %f\n";
  (* mat_flaten (List.hd nn.data.wl) |> mat_print ; *)
  mat_reshape (snd (List.hd train_data)) 28 28 |> mat_print;
  (* |> Mat.to_list |> List.length |> Printf.printf "Reshaped mat len %d\n"; *)

  (* if not @@ String.equal !save_file "" *)
  (* then save_to_json !save_file trained_nn.data; *)

  ()

let () =
  Unix.time () |> int_of_float |> Random.init ;
  Arg.parse speclist anon_fun usage_msg ;

  if (List.length !arch) = 0
  then 
       invalid_arg usage_msg
  else train
         !train_data_file
         !epoch_num

         [
           FullyConnected { ncount = 784;
             activation = sigmoid;
             derivative = sigmoid'
           };
           FullyConnected { ncount = 16;
             activation = sigmoid;
             derivative = sigmoid'
           };
           FullyConnected { ncount = 16;
             activation = sigmoid;
             derivative = sigmoid'
           } ;
           FullyConnected { ncount = 10;
             activation = sigmoid;
             derivative = sigmoid'
           }
         ]



 *)

let () =
  match Ocadl__Deep.test () with
  | Ok o -> ()
  | Error err -> Printf.eprintf "error: %s\n" err
