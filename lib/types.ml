type layer_meta =
  | FullyConnectedMeta of Fully_connected.meta
  | Conv3DMeta of Conv3D.meta
  | PoolingMeta of Pooling.meta
  | FlattenMeta of Flatten.meta
  | InputMeta of Input3D.meta 

type (_, _) layer_params =
  | FullyConnectedParams : Fully_connected.params ->
      (Fully_connected.input, Fully_connected.out)  layer_params

  | Conv3DParams : Conv3D.params ->
      (Conv3D.input, Conv3D.out) layer_params

  | PoolingParams : (Pooling.input, Pooling.out) layer_params
  | FlattenParams : (Flatten.input, Flatten.out) layer_params
  | Input3Params : (Input3D.input, Input3D.out) layer_params

type (_, _) layer =
  | FullyConnected  : Fully_connected.t ->
                      (Fully_connected.input, Fully_connected.out) layer

  | Conv3D          : Conv3D.t  -> (Conv3D.input, Conv3D.out) layer
  | Pooling         : Pooling.t -> (Pooling.input, Pooling.out) layer
  | Flatten         : Flatten.t -> (Flatten.input, Flatten.out) layer
  | Input3          : Input3D.t -> (Input3D.input, Input3D.out) layer

type zero = unit

type _ succ =
  | Succ : 'n -> 'n succ

type one = zero succ

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
