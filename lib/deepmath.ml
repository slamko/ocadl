type col =
  | Col of int

type row =
  | Row of int
  
module type Matrix_type = sig
  type 'a t

  val make : row -> col -> float -> float t

  val random : row -> col -> float t

  val zero : 'a t -> float t

  val of_array : row -> col -> 'a array -> 'a t

  val of_list : row -> col -> 'a list -> 'a t

  val row_mat_of_list : 'a list -> 'a t
  
  val size : 'a t -> int

  val get : row -> col -> 'a t -> 'a

  val set : row -> col -> 'a t -> 'a -> unit

  val reshape : row -> col -> 'a t -> 'a t

  val flatten3d : 'a t array -> 'a t

  val map : ('a -> 'b) -> 'a t -> 'b t

  val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val fold_right : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b
  
  val add : float t -> float t -> float t option
  
  val sub : float t -> float t -> float t option

  val mult : float t -> float t -> float t option

  val sum : float t -> float

  val convolve : float t -> stride:int -> float t -> float t

  val dim1 : 'a t -> row

  val dim2 : 'a t -> col

  val print : float t -> unit
  
end

module Matrix : Matrix_type = struct
   type 'a t = {
      matrix : 'a array;
      rows : row;
      cols : col;
    }

  type shape = {
      size : int;
      dim1 : row;
      dim2 : col;
    }
  let size mat = mat.matrix |> Array.length

  let get_shape mat =
    { size = size mat;
      dim1 = mat.rows;
      dim2 = mat.cols;
    }

  let make (Row rows) (Col cols) init_val =
    { matrix = Array.make (rows * cols) init_val ;
      rows = Row rows;
      cols = Col cols;
    }

  let create (Row rows) (Col cols) finit =
    { matrix = Array.init (rows * cols) finit;
      rows = Row rows;
      cols = Col cols;
    }

  let of_array rows cols matrix =
    { matrix; rows; cols }

  let get_row (Row row) = row

  let get_col (Col col) = col

  let get (Row row) (Col col) mat =
    Array.get mat.matrix @@ (row * get_row mat.rows) + col

  let get_raw row col mat =
    get (Row row) (Col col) mat

  let set (Row row) (Col col) mat value =
    Array.set mat.matrix ((row * get_row mat.rows) + col) value

  let set_raw row col mat value =
    set (Row row) (Col col) mat value
  
  let map proc mat =
    {
      matrix = Array.map proc mat.matrix;
      rows = mat.rows;
      cols = mat.cols;
    }

  let random row col =
    create row col (fun _ -> (Random.float 2. -. 1.))

  let zero mat =
    make mat.rows mat.cols 0.
  
  let of_list rows cols lst =
    let matrix = Array.of_list lst in
    { matrix; rows; cols; }

  let row_mat_of_list lst =
    let arr = Array.of_list lst in
    { matrix = arr;
      rows = Row 1;
      cols = Col (arr |> Array.length)
    }
 
  let print mat =
    Array.iteri (fun i value ->
        Printf.printf "%f" value;
        if i mod get_col mat.cols = 0
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

  let flatten3d mat_arr = 
    { matrix = Array.fold_left (fun acc mat ->
                   Array.append acc mat.matrix) [| |] mat_arr;
      rows = Row 1;
      cols = Col (Array.fold_left (fun acc mat ->
                      get_col mat.cols + acc) 0 mat_arr);
    }
   
  let sum mat =
    mat
    |> fold_left (fun value acc -> value +. acc) 0. 

  let mult mat1 mat2 =
    if get_col mat1.cols <> get_row mat2.rows
    then None
    else
      let res_mat =
        { matrix = Array.make (get_row mat1.rows * get_col mat2.cols) 0.;
          rows = mat1.rows;
          cols = mat2.cols;
        } in

      for r = 0 to get_row res_mat.rows
      do for ac = 0 to get_col mat1.cols
         do for c = 0 to get_col res_mat.cols
            do get_raw r ac mat1
               |> ( *. ) @@ get_raw ac c mat2
               |> ( +. ) @@ get_raw r c res_mat
               |> set_raw r c res_mat;
            done
         done
      done ;

      Some res_mat

  let convolve mat ~stride:stride kernel =
    (* let kern_arr = kernel |> Mat.to_array in *)
    let kern_rows = dim1 kernel |> get_row in
    let kern_cols = dim2 kernel |> get_col in
    let mat_rows = dim1 mat |> get_row in
    let mat_cols = dim2 mat |> get_col in

    let res_mat = make
                    (Row (mat_rows - kern_rows + 1))
                    (Col (mat_cols - kern_cols + 1)) 0. in

    let rec convolve_rec kernel stride mat r c =
        if r = mat_rows
        then ()
        else
        if c + kern_cols >= mat_cols
        then convolve_rec kernel stride mat (r + stride) 0
        else
            let dot_mat = Option.get @@ mult mat kernel in
            let sum = sum dot_mat in
            set_raw (r / stride) (c / stride) res_mat sum;
            convolve_rec kernel stride mat r (c + stride)
    in

    convolve_rec kernel stride mat 0 0;
    res_mat

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

 
let arr_print arr =
  arr |> Array.iter @@ Printf.printf "El: %f\n"

