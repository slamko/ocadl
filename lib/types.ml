open Lacaml.D

type train_data = (Mat.t * Mat.t) list

type weight_list = mat list
type bias_list = mat list

type activation = float -> float
type deriv = float -> float

type fully_connected_grad = {
    weight_mat : mat;
    bias_mat : mat;
  }

type fully_connected_type = {
    activation : activation;
    derivative : deriv;
    data : fully_connected_grad;
  }

type conv2d_grad = {
    kernels : mat list;
    bias_mat : mat;
  }

type conv2d_type = {
    padding : int;
    stride : int;
    data : conv2d_grad;
  }

type pooling_type = {
    fselect : mat -> float;
    stride : int;
  }

type pooling =
  | Data of pooling_type

type input_type = unit

type common = {
    ncount : int;
  }

type 'a layer_data = ('a * common)

type layer =
  | FullyConnected of fully_connected_type 
  | Conv2D of conv2d_type 
  | Pooling of pooling_type 
  | Input of input_type 

type layer_grad =
  | FullyConnectedGrad of fully_connected_grad
  | Conv2DGrad of conv2d_grad

type nnet = {
    layers : layer layer_data list;
  }

type nnet_grad = {
    layer_grads : layer_grad list;
  }

type feed_forward = {
    res : mat list;
    wl_ff : mat list;
    bl_ff : mat list
  }




