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


let () =
  time () |> int_of_float |> Random.init ;

  let train_data = read_train_data "data.csv" 1 28 in
  (* List.hd train_data |>  |> Mat.dim2 |> print_int ; *)
  (* let train_data = adder_data in *)
  let nn = make_nn [784; 16; 16; 16; 10] in
  (* nn.wl |> List.hd |> Mat.dim2 |> print_int; *)

  cost train_data nn |> Printf.printf "Cost: %f\n" ;

  let trained_nn = learn train_data 1000 nn in
  trained_nn |> cost train_data |> Printf.printf "Trained Cost %f\n";
  perform trained_nn train_data ;


  ()


