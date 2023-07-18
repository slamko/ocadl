open Lacaml.D

type train_data = (Mat.t * Mat.t) list

type weight_list = mat list
type bias_list = mat list

type activation = float -> float
type deriv = float -> float

type nnet_data = {
    wl : weight_list;
    bl : bias_list;
  }

type nnet = {
    data : nnet_data;
    activations : activation list;
    derivatives : deriv list
  }

type feed_forward = {
    res : mat list;
    wl_ff : mat list;
    bl_ff : mat list
  }




