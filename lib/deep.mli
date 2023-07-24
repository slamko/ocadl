open Types

(*

val learn : train_data -> ?epoch_num:int -> ?learning_rate:float ->
            ?batch_size:int -> nnet -> (nnet, string) result

val forward : mat list -> nnet -> feed_forward

val nn_gradient : nnet -> train_data -> nnet_params
 *)

val loss : train_data -> nnet -> (float, string) result

val test : unit -> unit


