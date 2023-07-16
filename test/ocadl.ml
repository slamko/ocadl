open Lacaml.D
open Unix
open Ocadl.Deep

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

let () =
  time () |> int_of_float |> Random.init ;

  let train_data = adder_data in
  let nn = make_nn [3; 9; 6; 2] in

  cost train_data nn |> Printf.printf "Cost: %f\n" ;

  let trained_nn = learn train_data 10000 nn in
  trained_nn |> cost train_data |> Printf.printf "Trained Cost %f\n";

  ()


