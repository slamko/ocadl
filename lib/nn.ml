open Lacaml.D
open Types
open Deepmath

let data arr =
  [| arr |] |> Mat.of_array

let one_data a =
  data [| a |]


let nn_print nn =
  print_string "\nNN print: \n" ;
  Printf.printf "Weights:\n" ;

  (* List.iter mat_print nn.wl ; *)
  Printf.printf "\nBiases:\n" ;
  ()
  (* List.iter mat_print nn.bl  *)

let list_print lst =
  List.iteri (fun i el -> Printf.printf "List element %d = %f\n" i el) lst

let list_split n lst =

  let rec split_rec n head tail =
    match n with
    | 0 -> (List.rev head, tail)
    | _ -> split_rec (n - 1) (List.hd tail :: head) (List.tl tail)
  in

  split_rec n [] lst

let build_nn meta_list param_list =
  { meta = { meta_list = meta_list; };
    params = { param_list = param_list; };
  }

let list_to_mat n lst =
  (* let lst_arr = lst |> Array.of_list in *)
  (* let data_len = Array.length lst_arr in *)
  (* let nrows = data_len / n in *)
  (* let res_arr = Array.make_matrix nrows n 0. in *)

  (* Array.iteri *)
    (* (fun i num -> *)
      (* let row = i / nrows in *)
      (* let col = i mod nrows in *)
      (* res_arr.(row).(col) <- num ; ) lst_arr ; *)

  [lst] |> Mat.of_list

let list_parse_train_data in_cols pair_list =
  List.map (fun (res_list, data_list) ->
      ([res_list] |> Mat.of_list, data_list |> list_to_mat in_cols)) pair_list

let read_train_data fname res_len in_cols =
  let csv = Csv.load fname in
  csv
  |> List.map @@ List.map float_of_string 
  |> List.map @@ list_split res_len
  |> List.map
       (fun (res_list, data_list) ->
         let res_mat = Array.make_matrix 1 10 0. in
         let col = (List.hd res_list |> int_of_float) in
         res_mat.(0).(col) <- 1. ;
         (res_mat |> Mat.of_array,
          data_list |> list_to_mat in_cols))
  (* |> List.iter (fun (res, inp) -> mat_print inp) *)

let to_json_list proc l =
  `List (l |> proc)

(*
let save_to_json fname nn =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in

  let mat_to_json_rec matl =
    `List
      (matl |>
         List.map @@
         to_json_list @@
             List.map @@
                to_json_list @@
                List.map
                   (fun num -> `Float num))
  in

  let json =
    `Assoc [
        ("weights",
         List.map Mat.to_list nn.wl
         |> mat_to_json_rec
        ) ;
        ("biases",
         List.map Mat.to_list nn.bl
         |> mat_to_json_rec
        )
      ] in

  let js_out = open_out fname in
  json |> Yojson.Basic.pretty_to_channel js_out ;
  close_out js_out

let restore_nn_from_json fname nn =
  let open Yojson.Basic.Util in
  let json =
    try Yojson.Basic.from_file fname
    with ex -> failwith "Invalid dump file." in

  let json_to_mat_list js_obj =
    js_obj
    |> to_list
    |> filter_list
    |> List.map filter_list
    |> List.map
       @@ List.map
       @@ List.map
            (fun maybe_num ->
              match to_float_option maybe_num with
              | Some num -> num
              | None -> failwith "Invalid dump file."
            )

    |> List.map Mat.of_list
  in    
  
  let weights =
    json
    |> member "weights"
    |> json_to_mat_list in

  let biases =
    json
    |> member "biases"
    |> json_to_mat_list  in

  { data = {
      wl = weights;
      bl = biases;
    };
    activations = nn.activations;
    derivatives = nn.derivatives
  }
  (* List.hd weights |> List.length |> print_int *)
 *)

let make_input shape =
  let in_layer = { layer = InputMeta ();
                   common = shape;
                 } in
  { meta =
      { meta_list = [in_layer] };
    params =
      { param_list = [] };
  }

let make_fully_connected ~ncount:ncount ~act:act ~deriv:deriv nn : nnet =
  let prev_ncount = (List.hd nn.meta.meta_list).common.ncount in
  
  let meta =
    FullyConnectedMeta
    { activation = act;
      derivative = deriv;
    }
  in

  let params =
    FullyConnectedParams
      { weight_mat = Mat.random prev_ncount ncount;
        bias_mat = Mat.random 1 ncount;
      } 
  in

  let common = { ncount = ncount } in
  let layer = { layer = meta;
                    common = common;
                  } in

  {
    meta = {
        meta_list = layer::nn.meta.meta_list;
      };
    params = {
        param_list = params::nn.params.param_list;
      };
  }


let make_nn (arch : nnet) : nnet =
  {
    meta = {
      meta_list = arch.meta.meta_list |> List.rev;
    };
    params = {
        param_list = arch.params.param_list |> List.rev;
      };
  }

let fully_connected_map proc layer =
  FullyConnectedParams
    { weight_mat = Mat.map proc layer.weight_mat;
      bias_mat = Mat.map proc layer.bias_mat;
    }

let conv2d_map proc layer =
  Conv2DParams
    { kernels = layer.kernels |> List.map @@ Mat.map proc;
      bias_mat = Mat.map proc layer.bias_mat
    }

let nn_params_map proc nn_params =
  List.map (function
      | FullyConnectedParams fc -> fully_connected_map proc fc
      | Conv2DParams cv -> conv2d_map proc cv)
    nn_params.param_list

let fully_connected_zero layer =
  FullyConnectedParams
    { weight_mat = mat_zero layer.weight_mat;
       bias_mat  = mat_zero layer.bias_mat;
    }

let conv2d_zero layer =
  Conv2DParams
    { kernels  = layer.kernels  |> List.map mat_zero ;
      bias_mat = layer.bias_mat |> mat_zero ;
    }
                    
let nn_params_zero nn_params =
   List.map (function
      | FullyConnectedParams fc -> fully_connected_zero fc
      | Conv2DParams cv -> conv2d_zero cv)
     nn_params.param_list

let get_data_input sample =
  snd sample

let get_data_out sample =
  fst sample
