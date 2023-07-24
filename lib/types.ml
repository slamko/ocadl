open Deepmath

type train_data = (mat * mat list) list

type weight_list = mat list
type bias_list = mat list

type activation = float -> float
type deriv = float -> float

type fully_connected_params = {
    weight_mat : mat;
    bias_mat : mat;
  }

type fully_connected_meta = {
    activation : activation;
    derivative : deriv;
  }

type conv2d_params = {
    kernels : mat list;
    bias_mat : mat;
  }

type conv2d_meta = {
    padding : int;
    stride : int;
  }

type pooling_meta = {
    fselect : float -> mat -> float;
    stride : int;
    filter_rows : row;
    filter_cols : col;
  }

type input_meta = unit

type common = {
    ncount : int;
  }

type layer_meta =
  | FullyConnectedMeta of fully_connected_meta 
  | Conv2DMeta of conv2d_meta 
  | PoolingMeta of pooling_meta 
  | InputMeta of input_meta 

type layer_meta_common = {
    common : common;
    layer  : layer_meta;
  }

type layer_params =
  | FullyConnectedParams of fully_connected_params
  | Conv2DParams of conv2d_params

type layer =
  | FullyConnected of (fully_connected_meta * fully_connected_params)
  | Conv2D of (conv2d_meta * conv2d_params)
  | Pooling of pooling_meta
  | Input 

type nnet_params = {
    param_list : layer_params list;
  }

type nnet_meta = {
    meta_list : layer_meta_common list;
  }

type nnet = {
    meta : nnet_meta;
    params : nnet_params;
  }

type ff_input_type =
  | Tensor2 of mat
  | Tensor3 of mat Mat.t
  | Tensor4 of mat Mat.t

let make_tens2 v = Tensor2 v
let make_tens3 v = Tensor3 v
let make_tens4 v = Tensor4 v

let (>>|) v f =
  match v with
  | Some value -> Some (f value)
  | None -> None

let (>>=) v f =
  match v with
  | Some value -> f value
  | None -> None

type feed_forward = {
    res : ff_input_type list;
    backprop_nn : nnet;
  }


