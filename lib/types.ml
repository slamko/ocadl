open Deepmath
open Ppxlib

type weight_list = mat list
[@@deriving show]

type bias_list = mat list
[@@deriving show]

type activation = float -> float
[@@deriving show]

type deriv = float -> float
[@@deriving show]

type fully_connected_params = {
    weight_mat : mat;
    bias_mat : mat;
  }
[@@deriving show]

type fully_connected_meta = {
    activation : activation;
    derivative : deriv;
  }
[@@deriving show]

type conv2d_params = {
    kernels : mat array;
    bias_mat : mat;
  }
[@@deriving show]

type conv2d_meta = {
    padding : int;
    stride : int;
  }
[@@deriving show]

type pooling_meta = {
    fselect : float -> float -> float;
    stride : int;
    filter_rows : row;
    filter_cols : col;
  }
[@@deriving show]

type input_meta = unit
[@@deriving show]

type common = {
    ncount : int;
  }
[@@deriving show]

type layer_meta =
  | FullyConnectedMeta of fully_connected_meta 
  | Conv2DMeta of conv2d_meta 
  | PoolingMeta of pooling_meta 
  | InputMeta of input_meta 
[@@deriving show]

type layer_params =
  | FullyConnectedParams of fully_connected_params
  | Conv2DParams of conv2d_params
  | PoolingParams
  | InputParams
[@@deriving show]

type layer =
  | FullyConnected of (fully_connected_meta * fully_connected_params)
  | Conv2D of (conv2d_meta * conv2d_params)
  | Pooling of pooling_meta
  | Input 
[@@deriving show]

type nnet_params = {
    param_list : layer_params list;
  }
[@@deriving show]

type layer_common = {
    common : common;
    layer  : layer;
  }
[@@deriving show]

type nnet = {
    layers : layer_common list;
  }
[@@deriving show]

type ff_input_type =
  | Tensor1 of float Mat.t
  | Tensor2 of float Mat.t
  | Tensor3 of float Mat.t Mat.t
  | Tensor4 of float Mat.t Mat.t

let make_tens1 v = Tensor1 v
let make_tens3 v = Tensor3 v
let make_tens4 v = Tensor4 v
let make_tens4 v = Tensor4 v

type feed_forward = {
    res : ff_input_type list;
    backprop_nn : nnet;
  }

type train_data = (ff_input_type * ff_input_type) list



