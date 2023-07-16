open Lacaml.D

type train_data = (Mat.t * Mat.t) list

type weight_list = mat list
type bias_list = mat list

type nnet = {
    wl : weight_list;
    bl : bias_list
  }


