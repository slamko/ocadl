open Types
open Deepmath

let data arr =
  arr |> Mat.of_array (Row 1) @@ Col (Array.length arr)

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

let list_parse_train_data in_cols pair_list =
  List.map (fun (res_list, data_list) ->
      (res_list |> Mat.of_list (Row 1) @@ Col (List.length res_list),
       data_list |> Mat.of_list in_cols)) pair_list

let read_mnist_train_data fname shape =
  let csv = Csv.load fname in
  csv
  |> List.map @@ List.map float_of_string 
  |> List.map @@ list_split 1 
  |> List.map
       (fun (res_list, data_list) ->
         let res_mat = Array.make 10 0. in
         let (Col col) = Mat.Col (List.hd res_list |> int_of_float) in
         Array.set res_mat col 1.;
         Tensor1 res_mat,
         Tensor2 (data_list
                  |> Mat.of_list
                       shape.Mat.dim1
                       shape.Mat.dim2))
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

let make_input3d shape = 
  let in_layer = Input3 { shape; } in
  Build_Cons (in_layer, Build_Nil)

(*
let ( |>> ) : type a b c n. ((a, b) layer -> (n succ, a, c) build_list) ->
                   
  = fun fbuilder layer_list ->
  let prev = match layer_list with
    | Build_Cons (lay, tail) -> lay
  in
  fbuilder prev layer_list
 *)

let make_fully_connected ~ncount ~act ~deriv layers =
  let open Mat in
  let prev_ncount =
    match layers with
    | Build_Cons (lay, _) ->
       (match lay with
        | FullyConnected (meta, _) -> meta.out_shape
        | Flatten meta -> meta.out_shape
       ) |> shape_size
  in

  let meta =
    { Fully_Connected.
      activation = act;
      derivative = deriv;
      out_shape  = make_shape (Row 1) (Col ncount);
    }
  in

  let params =
    { Fully_Connected.
      weight_mat = random (Row prev_ncount) (Col ncount);
      bias_mat = random (Row 1) (Col ncount);
    } 
  in

  let layer = FullyConnected (meta, params); in
  Build_Cons (layer, layers)


let make_conv2d ~kernel_shape ~kernel_num
      ~act ~deriv ~padding ~stride layers =
  let open Mat in

  let prev_shape =
    match layers with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv2D (meta, _) -> meta.out_shape
        | Pooling meta -> meta.out_shape
        | Input3 meta -> meta.shape
       )
  in

  let new_dim in_dim kern_dim =
    ((in_dim + (2 * padding) - kern_dim)
     / stride) + 1 in
  
  let out_shape =
    make_shape
      (Row (new_dim
              (get_row prev_shape.dim1)
              (get_row kernel_shape.dim1)))
      (Col (new_dim
              (get_col prev_shape.dim2)
              (get_col kernel_shape.dim2))) in

  
  let meta = {
      Conv2D.
      padding;
      stride;
      act;
      deriv;
      kernel_num;
      out_shape;
   } in

  let kernels = create (Row kernel_num) (Col prev_shape.dim3) 
                     (fun _ _ -> random_of_shape kernel_shape) in

  let params = {
      Conv2D.
      kernels;
      bias_mat = random (Row 1) kernels.cols
    } in

  let layer = Conv2D (meta, params) in
  Build_Cons(layer, layers)

let make_pooling ~filter_shape ~stride ~f ~fbp layers =
  let open Mat in

  let prev_shape =
    match layers with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv2D (meta, _) -> meta.out_shape
        | Pooling meta -> meta.out_shape
        | Input3 meta -> meta.shape
       )
  in

  let new_dim in_dim filt_dim =
    ((in_dim +  - filt_dim)
     / stride) + 1 in
  
  let meta = { Pooling.
               fselect = f;
               fderiv = fbp;
               stride;
               filter_shape;
               out_shape =
                 make_shape
                   (Row (new_dim
                           (get_row prev_shape.dim1)
                           (get_row filter_shape.dim1)))
                   (Col (new_dim
                           (get_col prev_shape.dim2)
                           (get_col filter_shape.dim2)))
             } in

  let layer = Pooling meta in
  Build_Cons (layer, layers)

let make_flatten layers =
  let open Mat in
  let prev_shape =
    match layers with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv2D (meta, _) -> meta.out_shape
        | Pooling meta -> meta.out_shape
        | Input3 meta -> meta.shape
       )
  in

  let meta = { Flatten.out_shape =
                 make_shape (Row 1) (Col (shape_size prev_shape)) } in
  let layer = Flatten meta in
  Build_Cons (layer, layers)

let rev_build_list blist =
  let rec rev_rec : type a b c n. (n, a, b) build_list ->
                                (b, c) ff_list ->
                                (a, c) ff_list
    = fun blist acc ->
    match blist with
    | Build_Nil -> acc
    | Build_Cons (lay, tail) ->
       rev_rec tail (FF_Cons (lay, acc))
  in

  rev_rec blist FF_Nil

let make_nn arch =
  { layers = rev_build_list arch }

let fully_connected_map proc layer =
    let open Fully_Connected in
    FullyConnectedParams {
      weight_mat = Mat.map proc layer.Fully_Connected.weight_mat;
      bias_mat = Mat.map proc layer.Fully_Connected.bias_mat;
    }

let conv2d_map proc layer =
  let open Conv2D in
  Conv2DParams {
    kernels = layer.kernels |> Mat.map @@ Mat.map proc;
    bias_mat = Mat.map proc layer.bias_mat
  }

let fully_connected_zero layer =
  let open Fully_Connected in
  FullyConnectedParams
    { weight_mat = Mat.zero layer.weight_mat;
       bias_mat  = Mat.zero layer.bias_mat;
    }

let conv2d_zero layer =
  let open Conv2D in
  Conv2DParams
    { kernels  = layer.kernels  |> Mat.map Mat.zero ;
      bias_mat = layer.bias_mat |> Mat.zero ;
    }


let param_map : type a b. (float -> float) ->
             (a, b) layer_params ->
             (a, b) layer_params =
  fun proc lay ->
  (match lay with
   | FullyConnectedParams fc -> fully_connected_map proc fc
   | Conv2DParams cn -> conv2d_map proc cn
   | empty -> empty
  )

let param_zero : type a b. 
             (a, b) layer_params ->
             (a, b) layer_params =
  fun lay ->
  (match lay with
   | FullyConnectedParams fc -> fully_connected_zero fc
   | Conv2DParams cn -> conv2d_zero cn
   | empty -> empty
  )

let nn_params_map proc nn =
  let rec nn_params_map : type a b c. (float -> float) ->
                             (a, b) param_list -> 
                             (a, b) param_list =
  fun proc nn_params ->
  match nn_params with
  | PL_Nil as nil -> nil
  | PL_Cons (lay, tail) ->
     let tail_map = nn_params_map proc tail in
     let new_lay = param_map proc lay in
     PL_Cons (new_lay, tail_map)
  in

  { param_list = nn_params_map proc nn } 
                   
let nn_params_apply proc nn1 nn2 =

  let rec apply_rec : type a b. (a, b) param_list ->
                           (a, b) param_list ->
                           (a, b) param_list
    = fun pl1 pl2 ->
    match pl1, pl2 with
    | PL_Nil, PL_Nil -> PL_Nil
    | PL_Cons (l1, t1), PL_Cons (l2, t2) -> 
       (match l1, l2 with
        | FullyConnectedParams fc1, FullyConnectedParams fc2 ->
           let new_lay =
             FullyConnectedParams {
                 weight_mat = proc fc1.weight_mat fc2.weight_mat;
                 bias_mat = proc fc1.bias_mat fc2.bias_mat;
               } in
           let tail = apply_rec t1 t2 in
           PL_Cons(new_lay, tail)
        | Conv2DParams cv1, Conv2DParams cv2 ->
           let new_lay =
             Conv2DParams {
               kernels = Mat.map2
                           (fun v1 v2 -> proc v1 v2)
                           cv1.kernels cv2.kernels;
               bias_mat = proc cv1.bias_mat cv2.bias_mat;
               } in
           let tail = apply_rec t1 t2 in
           PL_Cons (new_lay, tail)
        | Input3Params, Input3Params   ->
           PL_Cons (Input3Params,  apply_rec t1 t2)
        | PoolingParams, PoolingParams ->
           PL_Cons (PoolingParams, apply_rec t1 t2)
        | FlattenParams, FlattenParams ->
           PL_Cons (FlattenParams, apply_rec t1 t2)
        | _ -> failwith "The world fucked up"
       )
    | _ -> failwith "The world fucked up"
  in

  let param_list = apply_rec nn1.param_list nn2.param_list in
  { param_list }

let nn_apply_params proc nn params =

  let rec apply_rec : type a b. (a, b) ff_list ->
                           (a, b) param_list ->
                           (a, b) ff_list
    = fun pl1 pl2 ->
    match pl1, pl2 with
    | FF_Nil, PL_Nil -> FF_Nil
    | FF_Cons (lay, t1), PL_Cons (apply_param, t2) -> 
       (match lay, apply_param with
       | FullyConnected (meta, nn_param),
         FullyConnectedParams apply_param ->
          let open Mat in
          (* Printf.eprintf "pr %d pc %d\n" *)
          (* (get_row (dim1 nn_param.weight_mat)) *)
          (* (get_col (dim2 nn_param.weight_mat)) ; *)
          let wmat = proc nn_param.weight_mat apply_param.weight_mat in
          let bmat = proc nn_param.bias_mat   apply_param.bias_mat   in
          
          let new_lay =
            FullyConnected (meta, {
                  weight_mat = wmat;
                  bias_mat = bmat;
              }) in

          FF_Cons(new_lay, apply_rec t1 t2)
       | FullyConnected (_, _), _ ->
          failwith "nn apply params: Incompatible param list."
       | Conv2D (meta, nn_param), Conv2DParams apply_param ->
          let kernels = Mat.map2
                              (fun v1 v2 -> proc v1 v2)
                              nn_param.kernels apply_param.kernels in
          let bias_mat = proc nn_param.bias_mat apply_param.bias_mat in
             
          let new_lay =
            Conv2D (meta, {
                  kernels;
                  bias_mat;
              }) in
          FF_Cons(new_lay, apply_rec t1 t2)
       | Conv2D (_, _), _ -> 
          failwith "nn apply params: Incompatible param list."
       | Input3 _, Input3Params ->
           FF_Cons (lay, apply_rec t1 t2)
        | Pooling _, PoolingParams ->
           FF_Cons (lay, apply_rec t1 t2)
        | Flatten _, FlattenParams ->
           FF_Cons (lay, apply_rec t1 t2)
        | _ -> failwith "The world fucked up"
       )
    | _ -> failwith "nn apply params: Incompatible list types"
  in

  let layers = apply_rec nn.layers params.param_list in
  { layers }

let nn_params_zero nn_params =
  let rec params_zero : type a b. (a, b) param_list ->
                             (a, b) param_list
    = fun nn_params ->
    match nn_params with
    | PL_Nil -> PL_Nil
    | PL_Cons (lay, tail) ->
       let new_lay = param_zero lay in
       PL_Cons (new_lay, params_zero tail)
  in

  params_zero nn_params

