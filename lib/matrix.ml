open Common

type size =
  | Empty
  | Size of int

exception InvalidIndex
module type Tensor = sig
  type 'a t 

  val size : 'a t -> int

  val get : row -> col -> 'a t -> 'a

  val get : row -> col -> 'a t -> 'a

  val set : row -> col -> 'a t -> 'a -> unit

  val set_bind : row -> col -> 'a t -> 'a -> 'a t

  (* val set_option : row -> col -> 'a t -> 'a option -> 'a t option *)

  val reshape : row -> col -> 'a t -> 'a t

  val flatten3d : 'a t array -> 'a array

  val flatten : 'a t t -> 'a t

  val compare : ('a -> 'a -> bool) -> 'a t -> 'a t -> bool

  val compare_float : float t -> float t -> bool

  (* val submatrix : row -> col -> row -> col -> 'a t -> 'a t *)
  val scale : float -> float t -> float t

  val add_const : float -> float t -> float t

  val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val fold_right : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b

  val map : ('a -> 'b) -> 'a t -> 'b t

  val mapi : (row -> col -> 'a -> 'b) -> 'a t -> 'b t

  val iteri2 : (row -> col -> 'a -> 'b -> unit) -> 'a t -> 'b t -> unit

  val iter2 : ('a -> 'b -> unit) -> 'a t -> 'b t -> unit
 
  val iteri : (row -> col -> 'a -> unit) -> 'a t -> unit

  val foldi_left : (row -> col -> 'a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val foldi_right : (row -> col -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b

  val iter : ('a -> unit) -> 'a t -> unit
  
  val add : float t -> float t -> float t
  
  val sub : float t -> float t -> float t

  val sum : float t -> float

  val dim1 : 'a t -> row

  val dim2 : 'a t -> col

  val print : float t -> unit

   [@@deriving show]
end

module type Shaped = sig
  type 'a t

  val shape_match : 'a t -> 'b t -> unit
end

module TensorBase = struct
type 'a t = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    }

   [@@deriving show]
end

module Tensor (T : Shaped with type 'a t = 'a TensorBase.t) = struct
  include TensorBase
  
let size mat = get_row mat.rows |> ( * ) @@ get_col mat.cols

let get_size mat =
  match size mat with
  | 0 -> Empty
  | size -> Size size

let get_first_index mat =
   (get_row mat.start_row * mat.stride) + get_col mat.start_col
 
 let get_index row col mat =
   get_first_index mat + (row * mat.stride) + col
 
let get_res (Row row) (Col col) mat =
  if row >= get_row mat.rows
  then Error "get: Matrix row index out of bounds"
  else
    if col >= get_col mat.cols
    then Error "get: Matrix col index out of bounds"
    else
      Ok (get_index row col mat |> Array.get mat.matrix)

let get (Row row) (Col col) mat =
  if row >= get_row mat.rows
  then invalid_arg "get: Matrix row index out of bounds"
  else
    if col >= get_col mat.cols
    then invalid_arg "get: Matrix col index out of bounds"
    else
      get_index row col mat |> Array.get mat.matrix

let get_raw row col mat =
  get (Row row) (Col col) mat

let set_bind_res (Row row) (Col col) mat value =
  if row >= get_row mat.rows
  then Error "set: Matrix row index out of bounds"
  else
    if col >= get_col mat.cols
    then Error "set: Matrix col index out of bounds"
    else
      begin Array.set mat.matrix (get_index row col mat) value;
            Ok (mat) end

let set_res row col mat value =
  set_bind_res row col mat value 

let set_bind (Row row) (Col col) mat value =
  if row >= get_row mat.rows
  then invalid_arg "set: Matrix row index out of bounds"
  else
    if col >= get_col mat.cols
    then invalid_arg "set: Matrix col index out of bounds"
    else begin
        Array.set mat.matrix (get_index row col mat) value;
        mat end

let set row col mat value =
  set_bind row col mat value |> ignore

let get_first mat = get (Row 0) (Col 0) mat


let of_array rows cols matrix =
  if (get_row rows * get_col cols) > (matrix |> Array.length) ||
       (get_row rows) < 0 || (get_col cols) < 0
  then invalid_arg "Of array: Index out of bounds."
  else
    let (Col col) = cols in
    let stride = col in
    let start_row = Row 0 in
    let start_col = Col 0 in
    { matrix; rows; cols; start_row; start_col; stride }

let qto_array mat = mat.matrix

let of_array_size matrix =
  of_array (Row 1) (Col (Array.length matrix)) matrix

let make (Row rows) (Col cols) init_val =
  Array.make (rows * cols) init_val
  |> of_array (Row rows) (Col cols)

let qcopy mat =
  of_array mat.rows mat.cols (Array.copy mat.matrix)

let create (Row rows) (Col cols) finit =
  Array.init (rows * cols) (fun i ->
      finit (Row (i / cols)) (Col (i mod cols)))
  |> of_array (Row rows) (Col cols)

let empty () = of_array (Row 0) (Col 0) [| |]

let iter proc mat =
  for r = 0 to get_row mat.rows - 1
  do for c = 0 to get_col mat.cols - 1
     do proc @@ get (Row r) (Col c) mat;
     done
  done

let random row col =
  create row col (fun _ _ -> (Random.float 2. -. 1.))

let opt_iter proc mat =
  let rec iter_rec (Row r) (Col c) proc mat =
    if r >= get_row mat.rows
    then Ok ()
    else if c >= get_col mat.cols
    then iter_rec (Row (r + 1)) (Col 0) proc mat
    else 
      match proc @@ get_raw r c mat with
      | Ok _ -> 
         iter_rec (Row r) (Col (c + 1)) proc mat
      | Error err -> Error err
  in
  
  iter_rec (Row 0) (Col 0) proc mat

let iteri proc mat =
  for r = 0 to get_row mat.rows - 1
  do for c = 0 to get_col mat.cols - 1
     do proc (Row r) (Col c) @@ get (Row r) (Col c) mat;
     done
  done

let iteri3 proc mat1 mat2 mat3 =
  T.shape_match mat1 mat2;
  T.shape_match mat1 mat3;

  for r = 0 to row mat1.rows - 1
  do for c = 0 to get_col mat1.cols - 1
     do
       proc (Row r) (Col c)
         (get (Row r) (Col c) mat1)
         (get (Row r) (Col c) mat2)
         (get (Row r) (Col c) mat3)
     done
  done;
  ()

let iter3 proc mat1 mat2 mat3 =
  iteri3 (fun _ _ v1 v2 v3 -> proc v1 v2 v3) mat1 mat2 mat3

let iter2 proc mat1 mat2 =
  T.shape_match mat1 mat2 ;
  for r = 0 to get_row mat1.rows - 1
  do for c = 0 to get_col mat1.cols - 1
     do
       get (Row r) (Col c) mat2
       |> proc @@ get (Row r) (Col c) mat1 ;
     done
  done;
  ()

let iteri2 proc mat1 mat2 =
  T.shape_match mat1 mat2 ;
  for r = 0 to get_row mat1.rows - 1
  do for c = 0 to get_col mat1.cols - 1
     do get (Row r) (Col c) mat2
        |> proc (Row r) (Col c) @@ get (Row r) (Col c) mat1;
     done
  done;
  ()

let set_raw row col mat value =
  set (Row row) (Col col) mat value

let mapi3 proc mat1 mat2 mat3 =
  match size mat1 + size mat2 + size mat3 with
  | 0 -> empty ()
  | _ ->
     let res_mat = proc (Row 0) (Col 0)
                     (mat1 |> get_first)
                     (mat2 |> get_first)
                     (mat3 |> get_first)
                   |> make mat1.rows mat1.cols in
     
    iteri3 (fun r c value1 value2 value3 ->
        proc r c value1 value2 value3
        |> set r c res_mat)
      mat1 mat2 mat3 ;
     res_mat

let map3 proc mat1 mat2 mat3 =
  mapi3 (fun _ _ v1 v2 v3 -> proc v1 v2 v3) mat1 mat2 mat3

let mapi2 proc mat1 mat2 =
  match size mat1 + size mat2 with
  | 0 -> empty ()
  | _ ->
     let res_mat = proc (Row 0) (Col 0)
                     (mat1 |> get_first)
                     (mat2 |> get_first)
                   |> make mat1.rows mat1.cols in
     
     let _ = iteri2 (fun r c value1 value2  ->
                 proc r c value1 value2 |> set r c res_mat)
               mat1 mat2 in
     res_mat

let map2 proc mat1 mat2 =
  mapi2 (fun _ _ -> proc) mat1 mat2

let mapi proc mat =
  match size mat with
  | 0 -> empty ()
  | _ ->
     let res_mat = mat
                   |> get_first
                   |> proc (Row 0) (Col 0)
                   |> make mat.rows mat.cols in
     
     iteri (fun r c value1 ->
         proc r c value1
         |> set r c res_mat)
       mat;
     
     res_mat

let map proc mat =
  mapi (fun _ _ -> proc) mat

let opt_mapi proc mat =
  let* res_mat = proc (Row 0) (Col 0) (mat |> get_first)
                 >>| make mat.rows mat.cols in
  
  let rec map_rec (Row r) (Col c) proc mat =
    if r >= get_row mat.rows
    then Ok res_mat
    else if c >= get_col mat.cols
    then map_rec (Row (r + 1)) (Col 0) proc mat
    else 
      match proc (Row r) (Col c) @@ get_raw r c mat with
      | Ok value -> 
         set_raw r c res_mat value ;
         map_rec (Row r) (Col (c + 1)) proc mat
      | Error err -> Error err
  in
  
  map_rec (Row 0) (Col 0) proc mat

let opt_map proc mat =
  opt_mapi (fun _ _ value -> proc value) mat

exception NotEqual

let compare cmp mat1 mat2 =
  try 
    iter2 (fun v1 v2 ->
        if not @@ cmp v1 v2
        then raise NotEqual) mat1 mat2;
    true 
  with NotEqual -> false

let compare_float mat1 mat2 =
  compare (fun v1 v2 -> abs_float (v2 -. v1) < 0.0001) mat1 mat2

let scale value mat =
  mat |> map @@ ( *. ) value

let add_const value mat =
  mat |> map @@ ( +. ) value
let zero mat =
  make mat.rows mat.cols 0.

let of_list rows cols lst =
  let matrix = Array.of_list lst in
  let (Col col) = cols in
  let stride = col in
  let start_row = Row 0 in
  let start_col = Col 0 in
  
  { matrix; rows; cols; start_row; start_col; stride }

let print mat =
  iteri (fun (Row r) (Col c) value ->
      if c = 0
      then print_string "\n";
      
      if value >= 0.
      then Printf.printf " %.0f " value
      else Printf.printf "%.0f " value;
    ) mat ;
  
  print_string "\n"

let print_bind mat =
  print mat;
  mat

let add mat1 mat2 =
  map2 (+.) mat1 mat2

let sub mat1 mat2 =
  map2 (-.) mat1 mat2 

let dim1 mat =
  mat.rows

let dim2 mat =
  mat.cols

let reshape rows cols mat =
  of_array rows cols mat.matrix

let submatrix start_row start_col rows cols mat =
  if get_row start_row + get_row rows > get_row mat.rows
     || get_col start_col + get_col cols > get_col mat.cols
  then invalid_arg "Submatrix: Index out of bounds."
  else
  
  let res_arr = Array.make (get_row rows * get_col cols) mat.matrix.(0) in
  let stride (Col col) = col in
  let res_mat = { matrix = res_arr;
                  rows = rows;
                  cols = cols;
                  start_row = start_row;
                  start_col = start_col;
                  stride = stride cols;
                }
  in
  
  iteri (fun r c value -> set r c res_mat value) mat;
  res_mat

let fold_right proc mat init =
  let acc = ref init in
  iter (fun el -> acc := proc el !acc) mat;
  !acc

let fold_left proc init mat =
  let acc = ref init in
  iter (fun el -> acc := proc !acc el) mat;
  !acc

let foldi_right proc mat init =
  let acc = ref init in
  iteri (fun r c el -> acc := proc r c el !acc) mat;
  !acc

let foldi_left proc init mat =
  let acc = ref init in
  iteri (fun r c el -> acc := proc r c !acc el) mat;
  !acc

let fold_left2 proc init mat1 mat2 =
  T.shape_match mat1 mat2 ;
  let acc = ref init in
  let _ = iter2 (fun val1 val2 -> acc := proc !acc val1 val2) mat1 mat2 in
  !acc

let fold_right2 proc mat1 mat2 init =
  T.shape_match mat1 mat2 ;
  let acc = ref init in
  let _ = iter2 (fun val1 val2 -> acc := proc val1 val2 !acc) mat1 mat2 in
  !acc

let flatten (mat_mat : 'a t t) =
  match get_size mat_mat with
  | Empty -> empty ()
  | Size _ -> 
     let size = fold_left (fun acc mat ->
                    acc + size mat) 0 mat_mat in
     
     let first = mat_mat |> get_first |> get_first in
     let res_mat = make (Row 1) (Col size) first in
     let index = ref 0 in
     
     iter (fun  mat ->
         iter (fun value ->
             res_mat.matrix.(!index) <- value;
             index := !index + 1) mat) mat_mat;
     
     res_mat

let flatten2d mat =
  mat |> reshape (Row 1) @@ Col (size mat)

let flatten3d mat_arr = 
  match mat_arr with
  | [| |] -> [| |]
  | mat_arr ->
     mat_arr
     |> of_array_size
     |> flatten
     |> qto_array

let reshape3d base mat =
  let base_size = fold_left (fun acc m -> acc + size m) 0 base in

  if base_size <> size mat
  then failwith "Reshape3D: invalid matrix size.";

  let res_mat = make base.rows base.cols (get_first base) in
  
  foldi_left (fun r c index inner ->
      
      let subm = submatrix
                   (Row 0) (Col index)
                   (Row 1) (Col (size inner)) mat
                 |> reshape inner.rows inner.cols
      in

      set r c res_mat subm ;
      index + size inner - 1
    ) 0 base |> ignore;

  res_mat

let rotate180 mat =
  let mrows = get_row mat.rows in
  let mcols = get_col mat.cols in
  mapi (fun (Row r) (Col c) _ ->
      get_raw (mrows - r - 1) (mcols - c - 1) mat 
      (* |> set_raw r c mat; *)
      (* set_raw (mrows - r - 1) (mcols - c - 1) mat value *)
    ) mat

let sum mat =
  mat |> fold_left (+.) 0. 
  end

module type Matrix = sig
  include Tensor
  val make : row -> col -> float -> float t

  val create : row -> col -> (row -> col -> 'a) -> 'a t

  val random : row -> col -> float t

  val zero : 'a t -> float t

  val of_array : row -> col -> 'a array -> 'a t

  val of_list : row -> col -> 'a list -> 'a t

  val shadow_submatrix : row -> col -> row -> col -> 'a t -> 'a t

  val mult : float t -> float t -> float t

  (* val convolve : float t -> padding:int -> stride:int -> shape -> *)
                 (* float t -> float t *)

end

module MatShaped = struct

  type 'a t = 'a TensorBase.t

  open TensorBase

  let shape_match mat1 mat2 =
    if row mat1.rows <> row mat2.rows || col mat1.cols <> col mat2.cols
    then failwith "Matrix shapes do not match."
   
end

module MatBase = Tensor (MatShaped)

module Mat : Matrix = struct 
  include MatBase 
  open TensorBase

  let mult mat1 mat2 =
    if col mat1.cols <> row mat2.rows
    then invalid_arg "Mult: Matrix geometry does not match."
    else
      let res_mat =
      Array.make (get_row mat1.rows * get_col mat2.cols) 0.
      |> of_array mat1.rows mat2.cols
    in
    
    for r = 0 to get_row res_mat.rows - 1
    do for ac = 0 to get_col mat1.cols - 1 
       do for c = 0 to get_col res_mat.cols - 1 
          do get_raw r ac mat1
             |> ( *. ) @@ get_raw ac c mat2
             |> ( +. ) @@ get_raw r c res_mat
             |> set_raw r c res_mat;
          done
       done
    done ;
    
    res_mat

let hadamard mat1 mat2 =
  map2 ( *. ) mat1 mat2

let padded padding mat =
  if padding = 0
  then mat
  else
    let mat_rows = dim1 mat |> get_row in
    let mat_cols = dim2 mat |> get_col in
  
    create
      (Row (mat_rows + (2 * padding)))
      (Col (mat_cols + (2 * padding)))
      (fun (Row r) (Col c) ->
        if r >= padding && c >= padding
           && r < mat_rows + padding
           && c < mat_cols + padding
        then get_raw (r - padding) (c - padding) mat
        else 0.
      )

let shadow_submatrix start_row start_col rows cols mat =
  if get_row start_row + get_row rows > get_row mat.rows
     || get_col start_col + get_col cols > get_col mat.cols
  then invalid_arg "Shadow submatrix: Index out of bounds."
  else
    let res_mat = { matrix = mat.matrix;
                    rows = rows;
                    cols = cols;
                    start_row = start_row;
                    start_col = start_col;
                    stride = get_col mat.cols;
                  }
    in
    res_mat

let convolve mat ~padding ~stride out_shape kernel =
  (* let kern_arr = kernel |> Mat.to_array in *)
  let kern_rows = dim1 kernel |> get_row in
  let kern_cols = dim2 kernel |> get_col in
 
  let base = padded padding mat in
  let base_rows = dim1 base |> get_row in
  let base_cols = dim2 base |> get_col in
  let res_mat = zero_of_shape out_shape in

  (* Printf.printf "Convolve sizes: base = %d, res = %d, kernel = %d\n" *)
    (* (size base) (size res_mat) (size kernel); *)
  
  let rec convolve_rec kernel r c res_mat =
    if r + kern_rows > base_rows 
    then res_mat
    else
      if c + kern_cols > base_cols
      then convolve_rec kernel (r + stride) 0 res_mat
      else
        (* let a = 4 in *)
        (* Printf.eprintf "r: %d; c: %d\n" r c ; *)
        let submat = shadow_submatrix (Row r) (Col c)
                       kernel.rows kernel.cols base in

        shape_match kernel submat;
        let conv = fold_left2
                     (fun acc val1 val2 -> acc +. (val1 *. val2))
                     0. submat kernel in
        set_raw (r / stride) (c / stride) res_mat conv;
        convolve_rec kernel r (c + stride) res_mat
    in

    convolve_rec kernel 0 0 res_mat

end

module type Vector = sig
  include Tensor

  val make : col -> float -> float t

  val create : col -> (row -> col -> 'a) -> 'a t

  val random : col -> float t

  val of_array : col -> 'a array -> 'a t

  val of_list : col -> 'a list -> 'a t

  val get : col -> 'a t -> 'a 

end

module Vec : Vector = struct
  include Tensor

  let get col vec =
    Tensor.get (Row 0) col vec
end
