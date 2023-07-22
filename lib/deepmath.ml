
module type Matrix_type = sig
  type 'a t

  val make : int -> int -> float -> float t

  val random : int -> int -> float t

  val of_array : int -> int -> 'a array -> 'a t

  val of_list : int -> int -> 'a list -> 'a t

  val row_mat_of_list : 'a list -> 'a t
  
  val size : 'a t -> int

  val get : int -> int -> 'a t -> 'a

  val set : int -> int -> 'a t -> 'a -> unit

  val reshape : int -> int -> 'a t -> 'a t

  val map : ('a -> 'b) -> 'a t -> 'b t

  val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val fold_right : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b
  
  val add : float t -> float t -> float t option
  
  val sub : float t -> float t -> float t option

  val mult : float t -> float t -> float t option

  val dim1 : 'a t -> int

  val dim2 : 'a t -> int

  val print : float t -> unit
  
end

module Matrix : Matrix_type = struct
  type 'a t = {
      matrix : 'a array;
      rows : int;
      cols : int;
    }

  type shape = {
      size : int;
      dim1 : int;
      dim2 : int;
    }

  let size mat = mat.matrix |> Array.length

  let get_shape mat =
    { size = size mat;
      dim1 = mat.rows;
      dim2 = mat.cols;
    }

  let make rows cols init_val =
    { matrix = Array.make (rows * cols) init_val ;
      rows = rows;
      cols = cols;
    }

  let create rows cols finit =
    { matrix = Array.init (rows * cols) finit;
      rows = rows;
      cols = cols;
    }

  let of_array rows cols matrix =
    { matrix; rows; cols }

  let get row col mat =
    Array.get mat.matrix @@ (row * mat.rows) + col

  let set row col mat value =
    Array.set mat.matrix ((row * mat.rows) + col) value

  let map proc mat =
    {
      matrix = Array.map proc mat.matrix;
      rows = mat.rows;
      cols = mat.cols;
    }

  let random row col =
    create row col (fun _ -> (Random.float 2. -. 1.))
  
  let of_list rows cols lst =
    let matrix = Array.of_list lst in
    { matrix; rows; cols; }

  let row_mat_of_list lst =
    let arr = Array.of_list lst in
    { matrix = arr;
      rows = 1;
      cols = arr |> Array.length
    }

  let print mat =
    Array.iteri (fun i value ->
        Printf.printf "%f" value;
        if i mod mat.cols = 0
        then print_string "\n"
      ) mat.matrix

  let add mat1 mat2 =
    if (get_shape mat1 |> compare @@ get_shape mat2) <> 0
    then None
    else
      let res_arr = Array.map2 (+.) mat1.matrix mat2.matrix in
      Some { matrix = res_arr;
             rows = mat1.rows;
             cols = mat1.cols;
        }

  let sub mat1 mat2 =
    if (get_shape mat1 |> compare @@ get_shape mat2) <> 0
    then None
    else
      let res_arr = Array.map2 (-.) mat1.matrix mat2.matrix in
      Some { matrix = res_arr;
             rows = mat1.rows;
             cols = mat1.cols;
        }

  let dim1 mat =
    mat.rows

  let dim2 mat =
    mat.cols

  let reshape rows cols mat =
    { matrix = mat.matrix;
      rows = rows;
      cols = cols;
    }

  let fold_right proc mat init =
    Array.fold_right proc mat.matrix init
  
  let fold_left proc init mat =
    Array.fold_left proc init mat.matrix
  
  let mult mat1 mat2 =
    if mat1.cols <> mat2.rows
    then None
    else
      let res_mat =
        { matrix = Array.make (mat1.rows * mat2.cols) 0.;
          rows = mat1.rows;
          cols = mat2.cols;
        } in

      for r = 0 to res_mat.rows
      do for ac = 0 to mat1.cols
         do for c = 0 to res_mat.cols
            do get r ac mat1
               |> ( *. ) @@ get ac c mat2
               |> ( +. ) @@ get r c res_mat
               |> set r c res_mat;
            done
         done
      done ;

      Some res_mat
  
end

module Mat = Matrix
type mat = float Mat.t

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

let make_zero_mat_list mat_list =
  List.fold_right (fun mat mlist ->
      (Mat.make (Mat.dim1 mat) (Mat.dim2 mat) 0.) ::  mlist) mat_list []

let arr_get index arr =
  Array.get arr index

let mat_add_const cst mat =
  mat |> Mat.map @@ (+.) cst

let mat_scale cst mat =
  mat |> Mat.map @@ ( *. ) cst

let mat_list_fold_left proc mat_list =
  List.fold_left (fun mat acc ->
      proc mat acc) mat_list []

let mat_list_flaten mlist =
  List.fold_right (fun lst flat_list_acc ->
        List.fold_right (fun num acc ->
            num :: acc) lst flat_list_acc) mlist [] 

let mat_sum mat =
  mat
  |> Mat.fold_left (fun value acc -> value +. acc) 0. 

(* let mat_list_flaten mat_list = *)
  

let convolve mat ~stride:stride kernel =
  (* let kern_arr = kernel |> Mat.to_array in *)
  let kern_rows = Mat.dim1 kernel in
  let kern_cols = Mat.dim2 kernel in
  let mat_rows = Mat.dim1 mat in
  let mat_cols = Mat.dim2 mat in

  let res_mat = Mat.make
                  (mat_rows - kern_rows + 1)
                  (mat_cols - kern_cols + 1) 0. in

  let rec convolve_rec kernel stride mat r c =
    if r = mat_rows
    then ()
    else
      if c + kern_cols >= mat_cols
      then convolve_rec kernel stride mat (r + stride) 0
      else
        let dot_mat = Option.get @@ Mat.mult mat kernel in
        let sum = mat_sum dot_mat in
        Mat.set (r / stride) (c / stride) res_mat sum;
        convolve_rec kernel stride mat r (c + stride)
  in
  
  convolve_rec kernel stride mat 0 0;
  res_mat

let mat_zero mat =
  Mat.make
    (Mat.dim1 mat)
    (Mat.dim2 mat) 0.
  
let arr_print arr =
  arr |> Array.iter @@ Printf.printf "El: %f\n"

