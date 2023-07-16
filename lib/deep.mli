open Lacaml.D
open Types

val cost : train_data -> nnet -> float

val learn : train_data -> int -> nnet -> nnet
