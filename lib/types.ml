module FC = Fully_connected 

type (_, _) layer_params =
  | FullyConnectedParams : FC.params -> (FC.input, FC.out)  layer_params
  | Conv3DParams : Conv3D.params -> (Conv3D.input, Conv3D.out) layer_params
  | PoolingParams : (Pooling.input, Pooling.out) layer_params
  | FlattenParams : (Flatten.input, Flatten.out) layer_params
  | Input3Params : (Input3D.input, Input3D.out) layer_params
  | Conv2DParams : Conv2D.params -> (Conv2D.input, Conv2D.out) layer_params
  | Pooling2DParams : (Pooling2D.input, Pooling2D.out) layer_params
  | Flatten2DParams : (Flatten2D.input, Flatten2D.out) layer_params
  | Input2Params : (Input2D.input, Input2D.out) layer_params
  | Input1Params : (Input1D.input, Input1D.out) layer_params

type (_, _) layer =
  | FullyConnected  : FC.t -> (FC.input, FC.out) layer
  | Conv3D          : Conv3D.t  -> (Conv3D.input, Conv3D.out) layer
  | Pooling         : Pooling.t -> (Pooling.input, Pooling.out) layer
  | Flatten         : Flatten.t -> (Flatten.input, Flatten.out) layer
  | Input3          : Input3D.t -> (Input3D.input, Input3D.out) layer
  | Conv2D          : Conv2D.t  -> (Conv2D.input, Conv2D.out) layer
  | Pooling2D       : Pooling2D.t -> (Pooling2D.input, Pooling2D.out) layer
  | Flatten2D       : Flatten2D.t -> (Flatten2D.input, Flatten2D.out) layer
  | Input2          : Input2D.t -> (Input2D.input, Input2D.out) layer
  | Input1          : Input1D.t -> (Input1D.input, Input1D.out) layer

type zero = unit

type _ succ =
  | Succ : 'n -> 'n succ

type (_, _,_) build_list =
  | Build_Nil : (zero, 'a, 'a) build_list
  | Build_Cons : ('b, 'c) layer *
              ('n, 'a, 'b) build_list ->
            ('n succ, 'a, 'c) build_list

type (_, _) ff_list =
  | FF_Nil : ('a, 'a) ff_list
  | FF_Cons : ('a, 'b) layer *
              ('b, 'c) ff_list ->
            ('a, 'c) ff_list

type (_, _) param_list =
  | PL_Nil : ('a, 'a) param_list
  | PL_Cons : ('a, 'b) layer_params *
              ('b, 'c) param_list ->
            ('a, 'c) param_list

type (_, _) bp_param_list =
  | BPL_Nil : ('a, 'a) bp_param_list
  | BPL_Cons : ('b, 'c) layer_params *
                 (('a, 'b) layer_params, _) bp_param_list ->
               (('a, 'c) layer_params, _) bp_param_list

type (_, _) bp_list =
  | BP_Nil : (('a, 'a) layer * 'a * 'a, _) bp_list
  | BP_Cons : ((('b , 'c) layer * 'b * 'c) *
               (('a , 'b) layer * 'a * 'b, _) bp_list)
            -> (('a , 'c) layer * 'a * 'c, _) bp_list

type ('n, 'a, 'b) build_nn = {
    build_input : ('a, 'a) layer;
    build_list : ('n, 'a, 'b) build_list;
  }

type ('a, 'b) nnet_params = {
    param_list : ('a, 'b) param_list;
  }

type ('n, 'a, 'b) nnet = {
    input : ('a, 'a) layer;
    layers : ('a, 'b) ff_list;
    build_layers : ('n, 'a, 'b) build_list;
  }

type ('a, 'b, 'c) feed_forward = {
    bp_input : ('a, 'a) layer * 'a * 'a;
    bp_data : (('a, 'b) layer * 'a * 'b, 'c) bp_list
  }

type ('a, 'b) train_data = ('b * 'a) list

let get_data_input sample =
  snd sample

let get_data_out sample =
  fst sample
