open Common
open Tensor

module VectorT : MATRIXABLE = struct
  type ('a, 'b) mat = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    }

  type 'a t = ('a, 'a) mat


end

module MatrixT : MATRIXABLE = struct
  type ('a, 'b) t = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    } [@@deriving show]

end

module VectorBase = Tensor (VectorT)

module MatrixBase = Tensor (MatrixT)

module type DEF = sig
 type ('a, 'b) mat = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    } [@@deriving show]

end

module rec Vector : sig
  include DEF
  type v 

  type 'a t = ('a, v) mat

  val flatten3d : 'a t array -> 'a array

  val flatten : 'a t t -> 'a t

  val compare : ('a -> 'a -> bool) -> 'a t -> 'a t -> bool

  val compare_float : float t -> float t -> bool

  val scale : float -> float t -> float t

  val add_const : float -> float t -> float t

  val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val fold_right : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b

  val map : ('a -> 'b) -> 'a t -> 'b t

  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  
  val map3 : ('a -> 'b -> 'c -> 'd) -> 'a t -> 'b t -> 'c t -> 'd t

  val iter : ('a -> unit) -> 'a t -> unit
  
  val add : float t -> float t -> float t
  
  val sub : float t -> float t -> float t

  val sum : float t -> float

  val print : float t -> unit

  type shape = {
      dim1 : col
    }

  val size : 'a t -> int

  val get : row -> col -> 'a t -> 'a

  val get_first : 'a t -> 'a

  val get_first_opt : 'a t -> 'a option

  val get : col -> 'a t -> 'a

  val set : col -> 'a t -> 'a -> unit

  val set_bind : row -> col -> 'a t -> 'a -> 'a t

  (* val set_option : row -> col -> 'a t -> 'a option -> 'a t option *)

  val reshape : row -> col -> 'a t -> 'a t

  val mapi : (row -> col -> 'a -> 'b) -> 'a t -> 'b t

  val iteri2 : (row -> col -> 'a -> 'b -> unit) -> 'a t -> 'b t -> unit

  val iter2 : ('a -> 'b -> unit) -> 'a t -> 'b t -> unit
 
  val iteri : (row -> col -> 'a -> unit) -> 'a t -> unit

  val foldi_left : (row -> col -> 'a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val foldi_right : (row -> col -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b

  val create : col -> (row -> col -> 'a) -> 'a t

  val make : col -> 'a -> 'a t

  val random : col -> float t

  val of_array : col -> 'a array -> 'a t

  val of_list : col -> 'a list -> 'a t

  val to_mat : 'a t -> 'a Mat.t

  val get : col -> 'a t -> 'a 

  val dim1 : 'a t -> col

  val shape_size : shape -> int

  val get_shape : 'a t -> shape

  val make_shape : col -> shape

end = struct
  include VectorBase
  module Base = VectorBase

 type ('a, 'b) mat = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    } [@@deriving show]
  type v 

  type 'a t = ('a, v) mat


  type shape = {
      dim1 : col
    }

  let get_shape vec =
    { dim1 = vec.cols }

  let shape_size shape =
    col shape.dim1

  let make cols init =
    Base.make (Row 1) cols init

  let create cols finit =
    Base.create (Row 1) cols finit

  let random cols =
    Base.random (Row 1) cols

  let of_array cols arr =
    Base.of_array (Row 1) cols arr

  let of_list cols lst =
    Base.of_list (Row 1) cols lst

  let to_mat (vec : 'a t) : 'a Mat.t =
    Mat.of_array (Row 1) vec.cols vec.matrix

  let get col vec =
    get (Row 0) col vec

  let set col vec =
    set (Row 0) col vec

  let dim1 vec =
    vec.cols

  let make_shape dim1 =
    { dim1 }

end
and Mat : sig
  type shape = {
      dim1 : row;
      dim2 : col;
    }
 include DEF
  type v 

  type 'a t = ('a, v) mat

  val flatten3d : 'a t array -> 'a array

  val flatten : 'a t t -> 'a t

  val compare : ('a -> 'a -> bool) -> 'a t -> 'a t -> bool

  val compare_float : float t -> float t -> bool

  val scale : float -> float t -> float t

  val add_const : float -> float t -> float t

  val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val fold_right : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b

  val map : ('a -> 'b) -> 'a t -> 'b t

  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  
  val map3 : ('a -> 'b -> 'c -> 'd) -> 'a t -> 'b t -> 'c t -> 'd t

  val iter : ('a -> unit) -> 'a t -> unit
  
  val add : float t -> float t -> float t
  
  val sub : float t -> float t -> float t

  val sum : float t -> float

  val print : float t -> unit

  val make : row -> col -> 'a -> 'a t

  val create : row -> col -> (row -> col -> 'a) -> 'a t

  val random : row -> col -> float t

  val zero : 'a t -> float t

  val of_array : row -> col -> 'a array -> 'a t

  val of_list : row -> col -> 'a list -> 'a t

  val shape_match : 'a t -> 'b t -> unit

  val shape_size : shape -> int

  val make_shape : row -> col -> shape

  val rotate180 : 'a t -> 'a t

  val get_shape : 'a t -> shape

  val shape_zero : unit -> shape 

  val size : 'a t -> int

  val get : row -> col -> 'a t -> 'a

  val get_first : 'a t -> 'a

  val get_first_opt : 'a t -> 'a option

  val get : row -> col -> 'a t -> 'a

  val set : row -> col -> 'a t -> 'a -> unit

  val set_bind : row -> col -> 'a t -> 'a -> 'a t

  (* val set_option : row -> col -> 'a t -> 'a option -> 'a t option *)

  val reshape : row -> col -> 'a t -> 'a t

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

  val print : float t -> unit
  
  val random_of_shape : shape -> float t

  val shadow_submatrix : row -> col -> row -> col -> 'a t -> 'a t

  val mult : float t -> float t -> float t

  val convolve : float t -> padding:int -> stride:int -> shape ->
                 float t -> float t

  val dim1 : 'a t -> row

  val dim2 : 'a t -> col

  val to_vec: 'a t -> 'a Vector.t

end = struct 
  include MatrixBase

 type ('a, 'b) mat = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    } [@@deriving show]
  type v 

  type 'a t = ('a, v) mat

  let to_vec mat =
    Vector.of_array mat.cols mat.matrix

  let zero_of_shape shape =
    make shape.dim1 shape.dim2 0.

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


  let random_of_shape shape =
    create shape.dim1 shape.dim2 (fun _ _ -> (Random.float 2. -. 1.))

end
