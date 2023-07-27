
let ( let* ) o f =
  match o with
  | Error err -> Error err
  | Ok x -> f x

let ( let@ ) o f =
  match o with
  | Error err -> failwith err
  | Ok x -> f x

let (>>|) v f =
  match v with
  | Ok value -> Ok (f value)
  | Error err -> Error err

let (>>=) v f =
  match v with
  | Ok value -> f value
  | Error err -> Error err

let unwrap res =
  match res with
  | Ok res -> res
  | Error err -> failwith err

type 'a res = ('a, string) result

