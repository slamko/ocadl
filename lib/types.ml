open Deepmath
open Matrix
open Ppxlib

type weight_list = mat list
[@@deriving show]

type bias_list = mat list
[@@deriving show]

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
end

module Conv2D = struct 
type params = {
    kernels : mat Mat.t;
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
end

module Pooling = struct 
type meta = {
    fselect : float -> float -> float;
    stride : int;
    filter_shape : shape;
    out_shape : shape;
  }
[@@deriving show]

end

type input_meta = {
    shape : shape;
  }
[@@deriving show]

module Flatten = struct
type meta = {
    out_shape : shape;
  }
[@@deriving show]
end


type layer_meta =
  | FullyConnectedMeta of Fully_Connected.meta
  | Conv2DMeta of Conv2D.meta
  | PoolingMeta of Pooling.meta
  | FlattenMeta of Flatten.meta
  | InputMeta of input_meta 
[@@deriving show]

type layer_params =
  | FullyConnectedParams of Fully_Connected.params
  | Conv2DParams of Conv2D.params
  | PoolingParams
  | FlattenParams
  | InputParams
[@@deriving show]

type layer =
  | FullyConnected of (Fully_Connected.meta * Fully_Connected.params)
  | Conv2D of (Conv2D.meta * Conv2D.params)
  | Pooling of Pooling.meta
  | Flatten of Flatten.meta
  | Input of input_meta
[@@deriving show]

type nnet_params = {
    param_list : layer_params list;
  }
[@@deriving show]

(*
type layer_common = {
    common : common;
    layer  : layer;
  }
[@@deriving show]
*)

type nnet = {
    layers : layer list;
  }
[@@deriving show]

type ff_input_type =
  | Tensor1 of float Mat.t
  | Tensor2 of float Mat.t
  | Tensor3 of float Mat.t Mat.t
  | Tensor4 of float Mat.t Mat.t
[@@deriving show]

let make_tens1 v = Tensor1 v
let make_tens2 v = Tensor2 v
let make_tens3 v = Tensor3 v
let make_tens4 v = Tensor4 v

type feed_forward = {
    res : ff_input_type list;
    backprop_nn : nnet;
  }

type train_data = (ff_input_type * ff_input_type) list



