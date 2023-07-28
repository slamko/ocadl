open Common
open Matrix

module Mat = Matrix
type mat = float Mat.Mat.t

[@@deriving show]

let sigmoid (x : float) : float =
  1. /. (1. +. exp(-. x))

let sigmoid' activation =
  activation *. (1. -. activation)

let tanh (x : float) : float =
  ((exp(2. *. x) -. 1.0)  /. (exp(2. *. x) +. 1.))

let tanh' activation =
  1. -. (activation *. activation)

let relu x =
  if x > 0. then x else 0.

let relu' a =
  if a > 0. then 1. else 0. 

let pooling_max a b =
  if a > b then a else b

let pooling_avarage a b =
  (a +. b) /. 2.

let make_zero_mat_list mat_list =
  List.fold_right (fun mat mlist ->
      (Mat.make (Mat.dim1 mat) (Mat.dim2 mat) 0.) ::  mlist) mat_list []

let arr_get index arr =
  Array.get arr index
 
let arr_print arr =
  arr |> Array.iter @@ Printf.printf "El: %f\n"

let opt_to_bool = function
  | Some b -> b
  | None -> false

let bool_res_to_bool = function
  | Ok b -> b
  | Error err ->
     Printf.eprintf "error: %s\n" err ;
     false

let res_to_bool = function
  | Ok _ -> true
  | Error err ->
     Printf.eprintf "error: %s\n" err ;
     false

let%test "compare" =
  let m = Mat.random (Row 3) (Col 4) in
  Mat.compare_float m m

let m1 = [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1.;
            -1.; -0.9; -0.8; -0.7; -0.6; -0.5; -0.4; -0.3; -0.2; -0.1|]
        |> Mat.of_array (Row 4) (Col 5)

let%test "shadow_submatrix" =
  let open Mat in

  let test () =
    let m2 = of_array (Row 2) (Col 2) [| -0.7;-0.6;-0.2;-0.1|] in
    m1
    |> shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2) 
    |> compare_float m2
  in

  test ()

let%test "add" =
  let open Mat in
  let t () = 
    let m2 = of_array (Row 2) (Col 2) [| 0.1; 0.2; 0.3; 0.4|] in
    let res = of_array (Row 2) (Col 2) [| -0.6;-0.4;0.1;0.3|] in
    m1
    |> shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2)
    |> add m2
    |> compare_float res
  in
  t () 
    
let%test "sub" =
  let open Mat in

  let test () = 
    let m2 = of_array (Row 2) (Col 2) [| 0.1; 0.2; 0.3; 0.4|] in
    let res = of_array (Row 2) (Col 2) [| 0.8;0.8;0.5;0.5|] in
    
    m1
    |> shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2) 
    |> sub m2
    |> compare_float res
  in
  
  test () 

let%test "flatten" =
  let open Mat in

  let test() =
    let m4 = of_array (Row 1) (Col 2) [| -0.7; -0.6;|] in
    let m5 = of_array (Row 1) (Col 2) [| -0.2; -0.1;|] in
    let res = of_array (Row 2) (Col 1) [| m4; m5|]
              |> flatten
              |> reshape (Row 2) (Col 2)
    in
    
    m1
    |> shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2) 
    |> compare_float res
  in
  
  test() 

let%test "convolve" =
  let open Mat in
  let test () =
    let im = Array.init 9 (fun i -> float_of_int i +. 1.)
              |> of_array (Row 3) (Col 3) in
    let kern = Array.init 4 (fun i -> float_of_int i +. 10.)
              |> of_array (Row 2) (Col 2) in
    let res = [| 145.; 191.; 283.;329.|] |> of_array (Row 2) (Col 2) in
    convolve im ~stride:1 ~padding:0 (make_shape (Row 2) (Col 2)) kern
    |> compare_float res
  in

  test ()

let%test "mult" =
  let open Mat in

  let test () =
    let m2 = of_array (Row 2) (Col 1) [| 0.1;
                                          0.2; |] in
    let res = of_array (Row 2) (Col 1) [| -0.19;-0.04;|] in
    let m3 = (m1 |> shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2)) in

    mult m3 m2
    |> compare_float res 
  in

  test () 

