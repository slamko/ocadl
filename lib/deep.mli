open Lacaml.D
open Types

val cost : train_data -> nnet -> float

val learn : train_data -> ?epoch_num:int -> ?learning_rate:float ->
            ?batch_size:int -> nnet -> (nnet, string) result

val forward : mat -> nnet -> feed_forward

val nn_gradient : nnet -> train_data -> nnet


