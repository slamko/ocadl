open Lacaml.D
open Unix
open Ocadl.Deep
open Ocadl.Types
open Ocadl.Nn
open Ocadl.Deepmath

let ident_data =
  [
    ([| [|0.|] |] |> Mat.of_array , [| [|0.|] |] |> Mat.of_array ) ;
    ([| [|1.|] |] |> Mat.of_array , [| [|1.|] |] |> Mat.of_array ) ;
    ([| [|0.|] |] |> Mat.of_array , [| [|0.|] |] |> Mat.of_array ) ;
    ([| [|1.|] |] |> Mat.of_array , [| [|1.|] |] |> Mat.of_array )
  ]

let xor_data =
  [
    ([| [|0.|] |] |> Mat.of_array , [| [|0.; 0.|] |] |> Mat.of_array ) ;
    ([| [|1.|] |] |> Mat.of_array , [| [|0.; 1.|] |] |> Mat.of_array ) ;
    ([| [|1.|] |] |> Mat.of_array , [| [|1.; 0.|] |] |> Mat.of_array ) ;
    ([| [|0.|] |] |> Mat.of_array , [| [|1.; 1.|] |] |> Mat.of_array )
  ]


let adder_data =
  [
    ([| [|0.; 0.|] |] |> Mat.of_array , [| [|0.;0.; 0.|] |] |> Mat.of_array ) ;
    ([| [|0.; 1.|] |] |> Mat.of_array , [| [|0.;0.; 1.|] |] |> Mat.of_array ) ;
    ([| [|0.; 1.|] |] |> Mat.of_array , [| [|0.;1.; 0.|] |] |> Mat.of_array ) ;
    ([| [|1.; 0.|] |] |> Mat.of_array , [| [|0.;1.; 1.|] |] |> Mat.of_array ) ;
    ([| [|0.; 1.|] |] |> Mat.of_array , [| [|1.;0.; 0.|] |] |> Mat.of_array );
    ([| [|1.; 0.|] |] |> Mat.of_array , [| [|1.;0.; 1.|] |] |> Mat.of_array );
    ([| [|1.; 0.|] |] |> Mat.of_array , [| [|1.;1.; 0.|] |] |> Mat.of_array );
    ([| [|1.; 1.|] |] |> Mat.of_array , [| [|1.;1.; 1.|] |] |> Mat.of_array )
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

     perform nn t


let usage_msg = "ocadl -l <train_data_file> -s <save_file> -i <epoch_num> [<arch>] ... "
let epoch_num = ref 11
let train_data_file = ref ""
let save_file = ref ""
let arch = ref []

let anon_fun layer =
  arch := (int_of_string layer)::!arch

let speclist =
  [
    ("-i", Arg.Set_int epoch_num, "Learning iteration count") ;
    ("-l", Arg.Set_string train_data_file, "Training data file name") ;
    ("-s", Arg.Set_string save_file, "Json file to dump the NN state")
  ]

let train train_data_fname epochs nn_arch =
  let train_data = read_train_data train_data_fname 1 28 in
  (* List.hd train_data |>  |> Mat.dim2 |> print_int ; *)
  (* let train_data = adder_data in *)
  let nn =
    if Sys.file_exists !save_file
    then restore_nn_from_json !save_file
    else make_nn nn_arch
  in
  
  let trained_nn =
    match learn train_data ~epoch_num:epochs ~batch_size:2 nn with
    | Ok new_nn -> new_nn
    | Error err -> failwith err
  in

  (* perform trained_nn train_data ; *)
  cost train_data nn |> Printf.printf "Cost: %f\n" ;
  trained_nn |> cost train_data |> Printf.printf "Trained Cost %f\n";

  if not @@ String.equal !save_file ""
  then save_to_json !save_file trained_nn;

  ()

let () =
  Arg.parse speclist anon_fun usage_msg ;

  if String.equal !train_data_file ""
  then 
       invalid_arg usage_msg
  else if (List.length !arch) = 0
  then 
       invalid_arg usage_msg
  else train !train_data_file !epoch_num (List.rev !arch)
  


