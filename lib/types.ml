open Matrix
open Alias

type _ tensor =
  | Tensor1 : float matrix -> float matrix tensor
  | Tensor3 : float matrix matrix -> float matrix matrix tensor
(* [@@deriving show] *)

type activation = float -> float
[@@deriving show]

type deriv = float -> float
[@@deriving show]

module Fully_Connected = struct
type params = {
    weight_mat : mat;
    bias_mat : mat;
  }
[@@deriving show]

type meta = {
    activation : activation;
    derivative : deriv;
    out_shape : shape;
  }
[@@deriving show]

type input = float matrix tensor
type out = float matrix tensor

type t = meta * params
[@@deriving show]
end

module Conv2D = struct 
type params = {
    kernels : mat matrix;
    bias_mat : mat;
  }
[@@deriving show]

type meta = {
    padding : int;
    stride : int;
    act : activation;
    deriv : deriv;
    kernel_num : int;
    out_shape : shape;
  }
[@@deriving show]

type input = mat matrix tensor
type out = mat matrix tensor

type t = meta * params
[@@deriving show]
end

module Pooling = struct 
type meta = {
    fselect : float -> float -> float;
    fderiv : shape -> float -> mat -> mat -> unit;
    stride : int;
    filter_shape : shape;
    out_shape : shape;
  }
[@@deriving show]

type input = mat matrix tensor
type out = mat matrix tensor

type t = meta
[@@deriving show]
end

module Input3D = struct

type meta = {
    shape : shape;
  }
[@@deriving show]

type input = mat matrix tensor
type out = mat matrix tensor

type t = meta
end

module Flatten = struct
type meta = {
    out_shape : shape;
  }
[@@deriving show]

type input = float matrix matrix tensor
type out = float matrix tensor

type t = meta
end


type layer_meta =
  | FullyConnectedMeta of Fully_Connected.meta
  | Conv2DMeta of Conv2D.meta
  | PoolingMeta of Pooling.meta
  | FlattenMeta of Flatten.meta
  | InputMeta of Input3D.meta 
[@@deriving show]

type (_, _) layer_params =
  | FullyConnectedParams : Fully_Connected.params ->
      (Fully_Connected.input, Fully_Connected.out)  layer_params

  | Conv2DParams : Conv2D.params ->
      (Conv2D.input, Conv2D.out) layer_params

  | PoolingParams : (Pooling.input, Pooling.out) layer_params
  | FlattenParams : (Flatten.input, Flatten.out) layer_params
  | Input3Params : (Input3D.input, Input3D.out) layer_params

type (_, _) layer =
  | FullyConnected  : Fully_Connected.t ->
                      (Fully_Connected.input, Fully_Connected.out) layer

  | Conv2D          : Conv2D.t  -> (Conv2D.input, Conv2D.out) layer
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

let make_tens1 v = Tensor1 v
(* let make_tens2 v = Tensor2 v *)
let make_tens3 v = Tensor3 v
(* let make_tens4 v = Tensor4 v *)

type ('a, 'b, 'c) feed_forward = {
    bp_input : ('a, 'a) layer * 'a * 'a;
    bp_data : (('a, 'b) layer * 'a * 'b, 'c) bp_list
  }

type ('a, 'b) train_data = ('b * 'a) list

let get_data_input sample =
  snd sample

let get_data_out sample =
  fst sample
