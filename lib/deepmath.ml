open Common

module Mat = Matrix.Mat
module Vec = Matrix.Vector

type mat = float Mat.t
type vec = float Vec.t

type 'a matrix = 'a Mat.t
type 'a vector = 'a Vec.t

let sigmoid (x : float) : float =
  1. /. (1. +. exp(-. x))

let sigmoid' activation =
  activation *. (1. -. activation)

let tanh (x : float) : float =
  ((exp(2. *. x) -. 1.0)  /. (exp(2. *. x) +. 1.))

let tanh' activation =
  1. -. (activation *. activation)

let relu x =
  if x > 0.00001 then x else 0.

let relu' a =
  if a > 0.00001 then 1. else 0. 

let pooling_max a b =
  if a > b then a else b

let pooling_avarage a b =
  (a +. b) /. 2.

let pooling_max_deriv shape grad res_mat mat =
  let open Mat in
  let _, r, c =
    Mat.foldi_left (fun cur_r cur_c (acc, r, c) value ->
        if value > acc
        then (value, cur_r, cur_c)
        else (acc, r, c)) (0., Row 0, Col 0) mat
  in

  Mat.set r c res_mat grad

let pooling_avarage_deriv shape grad res_mat _ =
  let open Mat in
  let mat_size = shape_size shape |> float_of_int in
  iteri (fun r c _ ->
      set r c res_mat (grad /. mat_size))

let make_zero_mat_list mat_list =
  List.fold_right (fun mat mlist ->
      (Mat.make (Mat.dim1 mat) (Mat.dim2 mat) 0.) ::  mlist) mat_list []

let arr_get index arr =
  Array.get arr index
 
let arr_print arr =
  arr |> Array.iter @@ Printf.printf "El: %f\n"

let hdtl lst = List.tl lst |> List.hd

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

let%test "rotate180" =
  let open Mat in
  let arr = Array.init 9 (fun i -> float_of_int i +. 1.) in
  let im = arr |> of_array (Row 3) (Col 3) in
  let res = arr |> Array.to_list |> List.rev |> of_list (Row 3) (Col 3)  in

  rotate180 im
  |> compare_float res

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

