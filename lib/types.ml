open Lacaml.D

type train_data = (Mat.t * Mat.t) list

type weight_list = mat list
type bias_list = mat list

type nnet = {
    wl : weight_list;
    bl : bias_list
  }

type feed_forward = {
    res : mat list;
    wl_ff : mat list;
    bl_ff : mat list
  }




