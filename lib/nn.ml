open Types
open Alias
open Common
open Deepmath
open Tensor
open C.Functions

let one_data value =
  Vec.of_array [|value|]

let data arr =
  Vec.of_array arr

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

let read_mnist_train_data fname shape =
  let csv = Csv.load fname in

  csv
  |> List.map @@ List.map float_of_string 
  |> List.map @@ list_split 1 
  |> List.map
       (fun (res_list, data_list) ->
         let res_vec = Vec.make (Col 10) 0. in
         let (Col col) = Col (List.hd res_list |> int_of_float) in
         Vec.set (Col col) res_vec 1.;
         let inp =
           (data_list
            |> Mat.of_list
                 shape.Mat.dim1
                 shape.Mat.dim2) in

         (Tensor1 res_vec,
          Tensor2 inp))

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

(* let x = *)
  (* (Build_Cons (FullyConnected {FullyConnected}, (Build_Cons (Input3, Build_Nil)))) *)

let make_input1d shape = 
  let in_layer = Input1 { shape = Shape.make_shape_vec shape; } in
  { build_input = in_layer;
    build_list = Build_Cons (in_layer, Build_Nil);
  }

let make_input2d shape = 
  let in_layer = Input2 { shape = Shape.make_shape_mat shape; } in
  { build_input = in_layer;
    build_list = Build_Cons (in_layer, Build_Nil);
  }

let make_input3d shape = 
  let in_layer = Input3 { shape; } in
  { build_input = in_layer;
    build_list = Build_Cons (in_layer, Build_Nil);
  }

let make_fully_connected ~ncount ~actf layers =
  let prev_ncount =
    match layers.build_list with
    | Build_Cons (lay, _) ->
       (match lay with
        | FullyConnected (meta, _) -> meta.out_shape
        (* | Flatten meta -> meta.out_shape *)
        | Flatten2D meta -> meta.out_shape
        | Input1 meta -> meta.shape
       ) |> Shape.shape_size
  in

  let meta =
    { Fully_connected.
      activation = actf;
      out_shape  = Shape.make_shape_vec (Vec.make_shape (Col ncount));
    }
  in

  let params =
    { Fully_connected.
      weight_mat = Mat.random (Row prev_ncount) (Col ncount);
      bias_mat = Vec.random (Col ncount) ;
    } 
  in

  let layer = FullyConnected (meta, params); in
  { layers with build_list = Build_Cons (layer, layers.build_list) }

let make_conv2d ~kernel_shape ~act ~padding ~stride layers =
  let open Mat in

  let prev_shape =
    match layers.build_list with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv2D (meta, _) -> meta.out_shape
        | Pooling2D meta -> meta.out_shape
        | Input2 meta -> meta.shape
       )
  in

  let new_dim in_dim kern_dim =
    ((in_dim + (2 * padding) - kern_dim) / stride) + 1 in

  let Shape.ShapeMat(prev_image_shape) = prev_shape in
  
  let out_shape =
    Shape.make_shape_mat
      (Mat.make_shape
         (Row (new_dim
                 (get_row prev_image_shape.dim1)
                 (get_row kernel_shape.dim1)))
         (Col (new_dim
                 (get_col prev_image_shape.dim2)
                 (get_col kernel_shape.dim2)))) 
  in
  
  let meta = {
      Conv2D.
      padding;
      stride;
      act;
      out_shape;
   } in

  let kernels = Mat.random kernel_shape.dim1 kernel_shape.dim2
  in

  let params = {
      Conv2D.
      kernels;
      bias_mat = Vec.random (Col 1)
    } in

  let layer = Conv2D (meta, params) in
  { layers with build_list = Build_Cons(layer, layers.build_list) }

let make_pooling2d ~filter_shape ~stride ~f layers =
  let open Mat in

  let prev_shape =
    match layers.build_list with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv2D (meta, _) -> meta.out_shape
        | Pooling2D meta -> meta.out_shape
        | Input2 meta -> meta.shape
       )
  in

  let new_dim in_dim filt_dim =
    ((in_dim +  - filt_dim) / stride) + 1 in

  let Shape.ShapeMat(prev_image_shape) = prev_shape in
  
  let meta = { Pooling2D.
               fselect = f;
               stride;
               filter_shape = Shape.make_shape_mat filter_shape;
               out_shape =
                 Shape.make_shape_mat
                   (Mat.make_shape
                      (Row (new_dim
                              (get_row prev_image_shape.dim1)
                              (get_row filter_shape.dim1)))
                      (Col (new_dim
                              (get_col prev_image_shape.dim2)
                              (get_col filter_shape.dim2))))
             } in

  let layer = Pooling2D meta in
  {layers with build_list = Build_Cons (layer, layers.build_list) }


(*
let make_conv3d ~kernel_shape ~kernel_num
      ~act ~padding ~stride layers =
  let open Mat in

  let prev_shape =
    match layers.build_list with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv3D (meta, _) -> meta.out_shape
        | Pooling meta -> meta.out_shape
        | Input3 meta -> meta.shape
       )
  in

  let new_dim in_dim kern_dim =
    ((in_dim + (2 * padding) - kern_dim) / stride) + 1 in

  let Shape.ShapeMatVec(prev_image_shape) = prev_shape in
  
  let out_shape =
    Shape.make_shape_mat_vec
      (Mat3.make_shape
         (Row (new_dim
                 (get_row prev_image_shape.dim1)
                 (get_row kernel_shape.dim1)))
         (Col (new_dim
                 (get_col prev_image_shape.dim2)
                 (get_col kernel_shape.dim2)))
         (Col kernel_num)) 
  in
  
  let meta = {
      Conv3D.
      padding;
      stride;
      act;
      kernel_num;
      out_shape;
   } in

  let kernels =
    Mat3.random kernel_shape.dim1 kernel_shape.dim2 (Col kernel_num)
  in

  let params = {
      Conv3D.
      kernels;
      bias_mat = kernels.shape.dim3 |> Vec.random 
    } in

  let layer = Conv3D (meta, params) in
  { layers with build_list = Build_Cons(layer, layers.build_list) }

let make_pooling ~filter_shape ~stride ~f layers =
  let open Mat in

  let prev_shape =
    match layers.build_list with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv3D (meta, _) -> meta.out_shape
        | Pooling meta -> meta.out_shape
        | Input3 meta -> meta.shape
       )
  in

  let new_dim in_dim filt_dim =
    ((in_dim +  - filt_dim) / stride) + 1 in

  let Shape.ShapeMatVec(prev_image_shape) = prev_shape in
  
  let meta = { Pooling.
               fselect = f;
               stride;
               filter_shape = Shape.make_shape_mat filter_shape;
               out_shape =
                 Shape.make_shape_mat_vec
                   (Mat3.make_shape
                      (Row (new_dim
                              (get_row prev_image_shape.dim1)
                              (get_row filter_shape.dim1)))
                      (Col (new_dim
                              (get_col prev_image_shape.dim2)
                              (get_col filter_shape.dim2)))
                      (prev_image_shape.dim3))
             } in

  let layer = Pooling meta in
  {layers with build_list = Build_Cons (layer, layers.build_list) }

let make_flatten layers =
  let prev_image_shape =
    match layers.build_list with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv3D (meta, _) -> meta.out_shape
        | Pooling meta -> meta.out_shape
        | Input3 meta -> meta.shape
       )
  in

  let meta =
    { Flatten.
      out_shape =
        Shape.make_shape_vec
        @@ Vec.make_shape (Col (Shape.shape_size prev_image_shape))

    } in

  let layer = Flatten meta in
  { layers with build_list = Build_Cons (layer, layers.build_list) }

 *)

let make_flatten2d layers =
  let Shape.ShapeMat (prev_image_shape) =
    match layers.build_list with
    | Build_Cons (lay, _) ->
       (match lay with
        | Conv2D (meta, _) -> meta.out_shape
        | Input2 meta -> meta.shape
        | Pooling2D meta -> meta.out_shape
       )
  in

  let meta =
    { Flatten2D.
      out_shape =
        Shape.make_shape_vec
        @@ Vec.make_shape
             (Col
                (Mat.shape_size prev_image_shape))

    } in

  let layer = Flatten2D meta in
  { layers with build_list = Build_Cons (layer, layers.build_list) }
 
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

let make_nn : type a b n. (n succ, a, b) build_nn -> (n succ, a, b) nnet =
  fun arch ->
  { input = arch.build_input;
    layers = rev_build_list arch.build_list;
    build_layers = arch.build_list;
  }

let fully_connected_zero layer =
  let open Fully_connected in
  FullyConnectedParams
    { weight_mat = Mat.zero
                     layer.weight_mat.shape.dim1
                     layer.weight_mat.shape.dim2;

      bias_mat = Vec.zero layer.bias_mat.shape.dim1;
    }

let conv2d_zero layer =
  let open Conv2D in
  Conv2DParams
    { kernels  =
        Mat.zero
          layer.kernels.shape.dim1
          layer.kernels.shape.dim2;

      bias_mat = Vec.zero layer.bias_mat.shape.dim1;
    }

(*
let conv3d_zero layer =
  let open Conv3D in
  Conv3DParams
    { kernels  =
        Mat3.zero
          layer.kernels.shape.dim1
          layer.kernels.shape.dim2
          layer.kernels.shape.dim3;

      bias_mat = Vec.zero layer.bias_mat.shape.dim1;
    }

 *)

let param_zero : type a b. 
             (a, b) layer_params ->
             (a, b) layer_params =
  fun lay ->
  (match lay with
   | FullyConnectedParams fc -> fully_connected_zero fc
   (* | Conv3DParams cn -> conv3d_zero cn *)
   | Conv2DParams cn -> conv2d_zero cn
   | empty -> empty
  )

let layer_zero : type a b. 
             (a, b) layer ->
             (a, b) layer_params =
  fun lay ->
  (match lay with
   | FullyConnected (_, fc) -> fully_connected_zero fc
   (* | Conv3D (_, cn) -> conv3d_zero cn *)
   (* | Flatten _ -> FlattenParams *)
   (* | Pooling _ -> PoolingParams *)
   | Conv2D (_, cn) -> conv2d_zero cn
   | Flatten2D _ -> Flatten2DParams
   | Pooling2D _ -> Pooling2DParams
   | Input3 _ -> Input3Params
   | Input2 _ -> Input2Params
   | Input1 _ -> Input1Params
  )

let fully_connected_scale value layer =
    let open Fully_connected in

    FullyConnectedParams {
        weight_mat = mat_scale value layer.weight_mat ;
        bias_mat = vec_scale value layer.bias_mat ;
    }

let conv2d_scale value layer =
  let open Conv2D in
  Conv2DParams {
      kernels = mat_scale value layer.kernels ;
      bias_mat = vec_scale value layer.bias_mat ;
  }

let param_scale : type a b. float ->
             (a, b) layer_params ->
             (a, b) layer_params =
  fun value lay ->
  (match lay with
   | FullyConnectedParams fc -> fully_connected_scale value fc
   | Conv2DParams cn -> conv2d_scale value cn
   | empty -> empty
  )

let fully_connected_print params =
  let open Fully_connected in
  Printf.printf "\nFully connected\n Weight mat: \n%!" ;
  Mat.print params.weight_mat ;
  Printf.printf "Bias mat: \n%!" ;
  Vec.print params.bias_mat ;
  ()

let conv2d_print params =
  let open Conv2D in
  Printf.printf "\nConv2D\n Kernel mat: \n%!" ;
  Mat.print params.kernels ;
  Printf.printf "Bias mat: \n%!" ;
  Vec.print params.bias_mat ;
  ()


let layer_print : type a b. 
             (a, b) layer ->
             unit =
  function
   | FullyConnected (_, fc) ->
      fully_connected_print fc
   | Conv2D (_, cn) ->
      conv2d_print cn
   | _ -> ()

let nn_print nn =
  let rec rec_print : type a b. (a, b) ff_list -> unit
    = fun nn ->
    match nn with
    | FF_Nil -> ()
    | FF_Cons (lay, tail) ->
       layer_print lay ;
       rec_print tail
  in                            

  rec_print nn.layers

let nn_params_scale value nn =

  let rec nn_params_scale : type a b. float ->
                             (a, b) param_list -> 
                             (a, b) param_list =
    fun value nn_params ->
    match nn_params with
    | PL_Nil as nil -> nil
    | PL_Cons (lay, tail) ->
       let tail_map = nn_params_scale value tail in
       let new_lay = param_scale value lay in
       PL_Cons (new_lay, tail_map)
  in

  { param_list = nn_params_scale value nn.param_list } 
                   
(*
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
           let bias1_as_mat =
             (mat_flatten_bp (Row 1) fc1.bias_mat.shape.dim1 fc1.bias_mat) in

           let bias2_as_mat =
             (cc_mat_flatten_bp 1 (col fc2.bias_mat.shape.dim1)
                fc2.bias_mat.matrix) in

           let new_lay =
             FullyConnectedParams {
                 weight_mat = proc
                                fc1.weight_mat.matrix
                                fc2.weight_mat.matrix
                              |> Mat.create ;

                 bias_mat = proc bias1_as_mat bias2_as_mat
                            |> cc_mat_flatten
                            |> Vec.create
                 ;
               } in
           let tail = apply_rec t1 t2 in
           PL_Cons(new_lay, tail)

        (*
        | Conv3DParams cv1, Conv3DParams cv2 ->
           let bias1_as_mat =
             (cc_mat_flatten_bp 1 (col cv1.bias_mat.shape.dim1)
                cv1.bias_mat.matrix) in

           let bias2_as_mat =
             (cc_mat_flatten_bp 1 (col cv2.bias_mat.shape.dim1)
                cv2.bias_mat.matrix) in

           let new_lay =
             Conv3DParams {
               kernels = Vec.map2
                           (fun v1 v2 -> proc v1 v2)
                           cv1.kernels cv2.kernels;
               bias_mat = proc bias1_as_mat bias2_as_mat
                          |> cc_mat_flatten
                          |> Vec.create ;
               } in
           let tail = apply_rec t1 t2 in
           PL_Cons (new_lay, tail)
         *)
        | Input3Params, Input3Params   ->
           PL_Cons (Input3Params,  apply_rec t1 t2)
        | Input2Params, Input2Params   ->
           PL_Cons (Input2Params,  apply_rec t1 t2)
        | Input1Params, Input1Params   ->
           PL_Cons (Input1Params,  apply_rec t1 t2)
        | Flatten2DParams, Flatten2DParams   ->
           PL_Cons (Flatten2DParams,  apply_rec t1 t2)
        | PoolingParams, PoolingParams ->
           PL_Cons (PoolingParams, apply_rec t1 t2)
        | FlattenParams, FlattenParams ->
           PL_Cons (FlattenParams, apply_rec t1 t2)
        | _ -> failwith "Param list arity mismatch"
       )
    | _ -> failwith "The world fucked up"
  in

  let param_list = apply_rec nn1.param_list nn2.param_list in
  { param_list }
 *)

let nn_apply_params proc nn params =

  let rec apply_rec : type a b. (a, b) ff_list ->
                           (a, b) param_list ->
                           (a, b) ff_list
    = fun pl1 pl2 ->
    match pl1, pl2 with
    | FF_Nil , PL_Nil -> FF_Nil
    | FF_Cons (lay, t1), PL_Cons (apply_param, t2) -> 
       (match lay, apply_param with
       | FullyConnected (meta, nn_param),
         FullyConnectedParams apply_param ->
          let open Mat in

          let new_lay =
             FullyConnected (meta, {
                   weight_mat = mat_sub nn_param.weight_mat apply_param.weight_mat ;
                   bias_mat = vec_sub nn_param.bias_mat apply_param.bias_mat;
               }) in
           

          FF_Cons(new_lay, apply_rec t1 t2)

       | Conv2D (meta, nn_param), Conv2DParams apply_param ->
          let kernels = mat_sub nn_param.kernels apply_param.kernels in
          let bias_mat = vec_sub nn_param.bias_mat apply_param.bias_mat in

          let new_lay =
            Conv2D (meta, {
                  kernels;
                  bias_mat;
              }) in

          FF_Cons(new_lay, apply_rec t1 t2)

       | Input3 _, Input3Params ->
           FF_Cons (lay, apply_rec t1 t2)
       | Input2 _, Input2Params ->
           FF_Cons (lay, apply_rec t1 t2)
       | Input1 _, Input1Params ->
           FF_Cons (lay, apply_rec t1 t2)
       (* | Pooling _, PoolingParams -> *)
          (* FF_Cons (lay, apply_rec t1 t2) *)
       (* | Flatten _, FlattenParams -> *)
          (* FF_Cons (lay, apply_rec t1 t2) *)
       | Pooling2D _, Pooling2DParams ->
          FF_Cons (lay, apply_rec t1 t2)
       | Flatten2D _, Flatten2DParams ->
          FF_Cons (lay, apply_rec t1 t2)
       | _ -> failwith "Incompatible list types"
       )
    | _ -> failwith "nn apply params: Incompatible list lengths"
  in

  let layers = apply_rec nn.layers params.param_list in
  { nn with layers = layers }

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

  { param_list = params_zero nn_params.param_list }

let nn_zero_params nn =
  let rec params_zero : type a b. (a, b) ff_list ->
                             (a, b) param_list
    = fun nn_params ->
    match nn_params with
    | FF_Nil -> PL_Nil
    | FF_Cons (lay, tail) ->
       let new_lay = layer_zero lay in
       PL_Cons (new_lay, params_zero tail)
  in                            

  { param_list = params_zero nn.layers }
