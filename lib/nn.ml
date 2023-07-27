open Types
open Deepmath

let data arr =
  arr |> Mat.of_array (Row 1) @@ Col (Array.length arr) |> Result.get_ok

let one_data a =
  data [| a |]

let nn_print nn =
  print_string "\nNN print: \n" ;
  Printf.printf "Weights:\n" ;

  (* List.iter Mat.print nn.wl ; *)
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

let build_nn layer_list =
  { layers = layer_list; }

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

  lst |> Mat.of_list (Row 1) @@ Col (List.length lst)

let list_parse_train_data in_cols pair_list =
  List.map (fun (res_list, data_list) ->
      (res_list |> Mat.of_list (Row 1) @@ Col (List.length res_list),
       data_list |> list_to_mat in_cols)) pair_list

let read_train_data fname res_len in_cols =
  let csv = Csv.load fname in
  csv
  |> List.map @@ List.map float_of_string 
  |> List.map @@ list_split res_len
  |> List.map
       (fun (res_list, data_list) ->
         let res_mat = Mat.make (Row 1) (Col 10) 0. in
         let col = Col (List.hd res_list |> int_of_float) in
         Mat.set (Row 0) col res_mat 1.;
         Some (res_mat,
          data_list |> list_to_mat in_cols))
  (* |> List.iter (fun (res, inp) -> mat_print inp) *)

let a () =
  let a = 5 and b = 6 in
  a

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

let make_input rows cols =
  let in_layer = { layer = Input;
                   common = { ncount = rows * cols; };
                 } in
  { layers = [in_layer];
  }

let make_fully_connected ~ncount ~act ~deriv nn : nnet =
  let prev_ncount = Row (List.hd nn.layers).common.ncount in
  
  let meta =
    { activation = act;
      derivative = deriv;
    }
  in

  let params =
      { weight_mat = Mat.random prev_ncount (Col ncount);
        bias_mat = Mat.random (Row 1) (Col ncount);
      } 
  in

  let common = { ncount = ncount } in
  let layer = { layer = FullyConnected (meta, params);
                common = common;
              } in

  { layers = layer::nn.layers }

let make_nn (arch : nnet) : nnet =
  { layers = List.rev arch.layers }

let fully_connected_map proc layer =
  FullyConnectedParams
    { weight_mat = Mat.map proc layer.weight_mat;
      bias_mat = Mat.map proc layer.bias_mat;
    }

let conv2d_map proc layer =
  Conv2DParams
    { kernels = layer.kernels |> Array.map @@ Mat.map proc;
      bias_mat = Mat.map proc layer.bias_mat
    }

let nn_params_map proc nn_params =
  List.map (function
      | FullyConnectedParams fc -> fully_connected_map proc fc
      | Conv2DParams cv -> conv2d_map proc cv)
    nn_params.param_list

let fully_connected_zero layer =
  FullyConnectedParams
    { weight_mat = Mat.zero layer.weight_mat;
       bias_mat  = Mat.zero layer.bias_mat;
    }

let conv2d_zero layer =
  Conv2DParams
    { kernels  = layer.kernels  |> Array.map Mat.zero ;
      bias_mat = layer.bias_mat |> Mat.zero ;
    }

let nn_params_map proc nn =
  { param_list =
      List.map (fun l ->
          match l with
          | FullyConnectedParams fc ->
             FullyConnectedParams {
                 weight_mat = proc fc.weight_mat |> unwrap;
                 bias_mat = proc fc.bias_mat |> unwrap;
               }
          | Conv2DParams cv ->
             Conv2DParams {
                 kernels = Array.map (fun v -> proc v |> unwrap) cv.kernels;
                 bias_mat = proc cv.bias_mat |> unwrap;
               }
          | PoolingParams -> PoolingParams
          | InputParams -> InputParams
        ) nn.param_list ;
  } 

let nn_zero_params nn : nnet_params =
  let zero_layers = nn.layers
  |> List.map (fun common ->
      match common.layer with
      | FullyConnected (_, params) ->
           FullyConnectedParams
              { weight_mat = Mat.zero params.weight_mat;
                bias_mat = Mat.zero params.bias_mat;
           }
      | Conv2D (_, params) ->
           Conv2DParams
              { kernels = Array.map Mat.zero params.kernels;
                bias_mat = Mat.zero params.bias_mat;
              }
      | Pooling _ -> PoolingParams
      | Input -> InputParams
       ) in
  { param_list = zero_layers; }
                    
let%catch nn_params_apply proc nn1 nn2 =
  { param_list =
      List.map2 (fun l1 l2 ->
          match l1, l2 with
          | FullyConnectedParams fc1, FullyConnectedParams fc2 ->
             FullyConnectedParams {
                 weight_mat = proc fc1.weight_mat fc2.weight_mat |> unwrap;
                 bias_mat = proc fc1.bias_mat fc2.bias_mat |> unwrap;
               }
          | Conv2DParams cv1, Conv2DParams cv2 ->
             Conv2DParams {
                 kernels = Array.map2
                             (fun v1 v2 -> proc v1 v2 |> unwrap)
                             cv1.kernels cv2.kernels;
                 bias_mat = proc cv1.bias_mat cv2.bias_mat |> unwrap;
               }
          | InputParams, InputParams -> InputParams
          | PoolingParams, PoolingParams -> PoolingParams
          | _ -> failwith "nn apply: Param lists do not match."
          ) nn1.param_list nn2.param_list;
      }

let%catch nn_apply_params proc nn params =
  { layers =
      List.map2
        (fun lay apply_param ->
          match lay.layer, apply_param with
          | FullyConnected (meta, nn_param),
            FullyConnectedParams apply_param ->
             let open Mat in
             (* Printf.eprintf "pr %d pc %d\n" *)
               (* (get_row (dim1 nn_param.weight_mat)) *)
               (* (get_col (dim2 nn_param.weight_mat)) ; *)
             let@ wmat = proc nn_param.weight_mat apply_param.weight_mat in
             let@ bmat = proc nn_param.bias_mat   apply_param.bias_mat   in

             { common = lay.common;
               layer = FullyConnected
                         (meta, {
                              weight_mat = wmat;
                              bias_mat = bmat;
                         });
             }
          | FullyConnected (_, _), _ ->
             failwith "nn apply params: Incompatible param list."

          | Conv2D (meta, nn_param), Conv2DParams apply_param ->
             let kernels = Array.map2
                              (fun v1 v2 -> proc v1 v2 |> unwrap)
                              nn_param.kernels apply_param.kernels in
             let@ bias_mat = proc nn_param.bias_mat apply_param.bias_mat in

             { common = lay.common;
               layer = Conv2D
                         (meta, {
                            kernels;
                            bias_mat;
                         });
             }
      
          | Conv2D (_, _), _ -> 
             failwith "nn apply params: Incompatible param list."
          | _ -> lay

      ) nn.layers params.param_list
  }
                    
let nn_params_zero nn_params =
   List.map (function
      | FullyConnectedParams fc -> fully_connected_zero fc
      | Conv2DParams cv -> conv2d_zero cv
      | PoolingParams -> PoolingParams
      | InputParams -> InputParams
     )
     nn_params.param_list

let get_data_input sample =
  snd sample

let get_data_out sample =
  fst sample
