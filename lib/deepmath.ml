let ( let* ) o f =
  match o with
  | Error err -> Error err
  | Ok x -> f x

let (>>|) v f =
  match v with
  | Some value -> Some (f value)
  | None -> None

let (>>=) v f =
  match v with
  | Some value -> f value
  | None -> None

type col =
  | Col of int

type row =
  | Row of int
  
module type Matrix_type = sig
  type 'a t

  exception InvalidIndex

  val make : row -> col -> float -> float t

  val create : row -> col -> (row -> col -> 'a) -> 'a t

  val random : row -> col -> float t

  val zero : 'a t -> float t

  val of_array : row -> col -> 'a array -> 'a t option

  val of_list : row -> col -> 'a list -> 'a t

  val size : 'a t -> int

  val get : row -> col -> 'a t -> 'a

  val set : row -> col -> 'a t -> 'a -> unit

  val set_bind : row -> col -> 'a t -> 'a -> 'a t

  val set_option : row -> col -> 'a t -> 'a option -> 'a t option

  val reshape : row -> col -> 'a t -> 'a t option

  val flatten3d : 'a t array -> 'a t option

  val flatten : 'a t t -> 'a t option

  val compare : ('a -> 'b -> bool) -> 'a t -> 'b t -> bool

  val compare_float : float t -> float t -> bool

  (* val submatrix : row -> col -> row -> col -> 'a t -> 'a t *)

  val shadow_submatrix : row -> col -> row -> col -> 'a t -> 'a t option


  val scale : float -> float t -> float t

  val add_const : float -> float t -> float t

  val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b t -> 'a

  val fold_right : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b

  val map : ('a -> 'b) -> 'a t -> 'b t

  val mapi : (row -> col -> 'a -> 'b) -> 'a t -> 'b t

  val iteri2 : (row -> col -> 'a -> 'b -> unit) -> 'a t -> 'b t
               -> (unit, unit) result

  val iter2 : ('a -> 'b -> unit) -> 'a t -> 'b t
              -> (unit, unit) result
 
  val iteri : (row -> col -> 'a -> unit) -> 'a t -> unit

  val iter : ('a -> unit) -> 'a t -> unit
  
  val add : float t -> float t -> float t option
  
  val sub : float t -> float t -> float t option

  val mult : float t -> float t -> float t option

  val sum : float t -> float

  val convolve : float t -> stride:int -> float t -> float t option

  val dim1 : 'a t -> row

  val dim2 : 'a t -> col

  val get_row : row -> int

  val get_col : col -> int

  val print : float t -> unit
  
end

module Matrix : Matrix_type = struct
   type 'a t = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    }

  exception InvalidIndex

  type shape = {
      size : int;
      dim1 : row;
      dim2 : col;
    }

  let get_row (Row row) = row

  let get_col (Col col) = col

  let size mat = get_row mat.rows |> ( * ) @@  get_col mat.cols

  let get_shape mat =
    { size = size mat;
      dim1 = mat.rows;
      dim2 = mat.cols;
    }

  let shape_match mat1 mat2 =
    let shape = get_shape mat2 in
    match get_shape mat1 |> compare shape with
    | 0 -> Ok shape
    | _ -> Error "Matrix shapes do not match" 

  let of_array rows cols matrix =
    if (get_row rows * get_col cols) > (matrix |> Array.length)
    then None
    else
      let (Col col) = cols in
      let stride = col in
      let start_row = Row 0 in
      let start_col = Col 0 in
      Some { matrix; rows; cols; start_row; start_col; stride }
  
  let make (Row rows) (Col cols) init_val =
    Array.make (rows * cols) init_val
    |> of_array (Row rows) (Col cols)
    |> Option.get
    
  let create (Row rows) (Col cols) finit =
    Array.init (rows * cols) (fun i ->
        finit (Row (i / rows)) (Col (i mod cols)))
    |> of_array (Row rows) (Col cols)
    |> Option.get

  let get_first mat =
    (get_row mat.start_row * mat.stride) + get_col mat.start_col

  let get_index row col mat =
    get_first mat + (row * mat.stride) + col

  let get (Row row) (Col col) mat =
    if row >= get_row mat.rows
    then failwith "matrix row index out of bounds";

    if col >= get_col mat.cols
    then failwith "matrix col index out of bounds";

    get_index row col mat |> Array.get mat.matrix

  let get_raw row col mat =
    get (Row row) (Col col) mat

  let set_bind (Row row) (Col col) mat value =
    if row >= get_row mat.rows
    then failwith "matrix row index out of bounds";

    if col >= get_col mat.cols
    then failwith "matrix col index out of bounds";

    Array.set mat.matrix (get_index row col mat) value;
    mat

  let set row col mat value =
    set_bind row col mat value |> ignore ;
    ()

  let set_option row col mat value =
    value >>| set_bind row col mat
    
  let iter proc mat =
    for r = 0 to get_row mat.rows - 1
    do for c = 0 to get_col mat.cols - 1
       do proc @@ get (Row r) (Col c) mat;
       done
    done

  let iteri proc mat =
    for r = 0 to get_row mat.rows - 1
    do for c = 0 to get_col mat.cols - 1
       do proc (Row r) (Col c) @@ get (Row r) (Col c) mat;
       done
    done

  let iter2 proc mat1 mat2 =
    let* _ = shape_match mat1 mat2 in
       for r = 0 to get_row mat1.rows - 1
       do for c = 0 to get_col mat1.cols - 1
          do
            get (Row r) (Col c) mat2
            |> proc @@ get (Row r) (Col c) mat1 ;
          done
       done;
       Ok ()

  let iteri2 proc mat1 mat2 =
    match shape_match mat1 mat2 with
    | Some _ ->
       for r = 0 to get_row mat1.rows - 1
       do for c = 0 to get_col mat1.cols - 1
          do get (Row r) (Col c) mat2
             |> proc (Row r) (Col c) @@ get (Row r) (Col c) mat1;
          done
       done;
       Ok ()
    | None -> Error ()

  let set_raw row col mat value =
    set (Row row) (Col col) mat value

  let mapi2 proc mat1 mat2 =
    let res_mat = proc (Row 0) (Col 0)
                    (mat1 |> get (Row 0) (Col 0))
                    (mat2 |> get (Row 0) (Col 0))
                  |> make mat1.rows mat1.cols in

    begin match iteri2 (fun r c value1 value2  ->
                    proc r c value1 value2 |> set r c res_mat)
                    mat1 mat2 with
    | Ok _ -> Some res_mat
    | Error _ -> None end

  let map2 proc mat1 mat2 =
    mapi2 (fun _ _ -> proc) mat1 mat2
    
  let mapi proc mat =
    let res_mat = proc (Row 0) (Col 0) (mat |> get (Row 0) (Col 0))
                |> make mat.rows mat.cols in

    iteri (fun r c value1 ->
        proc r c value1 |> set r c res_mat)
      mat;

    res_mat
    
  let map proc mat =
    mapi (fun _ _ -> proc) mat
  
  exception NotEqual

  let compare cmp mat1 mat2 =
    try begin
        match iter2 (fun v1 v2 ->
                  if not @@ cmp v1 v2
                  then raise NotEqual) mat1 mat2
        with
        | Ok _ -> true
        | Error _ -> false end
    with NotEqual -> false

  let compare_float mat1 mat2 =
    compare (fun v1 v2 -> abs_float (v2 -. v1) < 0.0001) mat1 mat2
    
  let scale value mat =
    mat |> map @@ ( *. ) value

  let add_const value mat =
    mat |> map @@ ( +. ) value

  let random row col =
    create row col (fun _ _ -> (Random.float 2. -. 1.))

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
        then Printf.printf " %f   " value
        else Printf.printf "%f   " value;
      ) mat ;
      
    print_string "\n"
      

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

  let shadow_submatrix start_row start_col rows cols mat =
    if get_row start_row + get_row rows > get_row mat.rows
    then None
    else if get_col start_col + get_col cols > get_col mat.cols
    then None
    else
      let res_mat = { matrix = mat.matrix;
                      rows = rows;
                      cols = cols;
                      start_row = start_row;
                      start_col = start_col;
                      stride = get_col mat.cols;
                    }
      in
      Some res_mat

  let fold_right proc mat init =
    let acc = ref init in
    iter (fun el -> acc := proc el !acc) mat;
    !acc
  
  let fold_left proc init mat =
    let acc = ref init in
    iter (fun el -> acc := proc !acc el) mat;
    !acc

  let fold_left2 proc init mat1 mat2 =
    let* _ = shape_match mat1 mat2 in
    let acc = ref init in
    match iter2 (fun val1 val2-> acc := proc !acc val1 val2) mat1 mat2 with
    | Ok _ -> Some !acc
    | Error _ -> None

  let flatten3d mat_arr = 
    let cols = Array.fold_left (fun acc mat ->
                      get_col mat.cols + acc) 0 mat_arr in
    Array.fold_left
      (fun acc mat -> Array.append acc mat.matrix) [| |] mat_arr
    |> of_array (Row 1) (Col cols)

  let flatten mat_mat =
    flatten3d mat_mat.matrix
   
  let sum mat =
    mat
    |> fold_left (fun value acc -> value +. acc) 0. 

  let mult mat1 mat2 =
    if get_col mat1.cols <> get_row mat2.rows
    then None
    else
      let res_mat =
        Array.make (get_row mat1.rows * get_col mat2.cols) 0.
        |> of_array mat1.rows mat2.cols
      in

      match res_mat with
      | Some res_mat ->
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
         
         Some res_mat
      | None -> None

  let convolve mat ~stride:stride kernel =
    (* let kern_arr = kernel |> Mat.to_array in *)
    let kern_rows = dim1 kernel |> get_row in
    let kern_cols = dim2 kernel |> get_col in
    let mat_rows = dim1 mat |> get_row in
    let mat_cols = dim2 mat |> get_col in

    let res_mat = make
                    (Row (mat_rows - kern_rows + 1))
                    (Col (mat_cols - kern_cols + 1)) 0. in

    let rec convolve_rec kernel stride mat r c res_mat =
        if r = mat_rows
        then Some res_mat
        else
        if c + kern_cols >= mat_cols
        then convolve_rec kernel stride mat (r + stride) 0 res_mat
        else
          let* submat = shadow_submatrix (Row r) (Col c)
                          kernel.rows kernel.cols mat in
          let* _ = shape_match kernel submat in
          let* conv = fold_left2
                        (fun acc val1 val2 -> acc +. (val1 *. val2))
                        0. submat kernel in
          set_raw (r / stride) (c / stride) res_mat conv;
          convolve_rec kernel stride mat r (c + stride) res_mat
    in

    convolve_rec kernel stride mat 0 0 res_mat
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

let mat_list_fold_left proc mat_list =
  List.fold_left (fun mat acc ->
      proc mat acc) mat_list []

let mat_list_flaten mlist =
  List.fold_right (fun lst flat_list_acc ->
        List.fold_right (fun num acc ->
            num :: acc) lst flat_list_acc) mlist [] 

 
let arr_print arr =
  arr |> Array.iter @@ Printf.printf "El: %f\n"

let opt_to_bool = function
  | Some b -> b
  | None -> false

let%test "compare" =
  let m = Mat.random (Row 3) (Col 4) in
  Mat.compare_float m m

let m1 = [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1.;
            -1.; -0.9; -0.8; -0.7; -0.6; -0.5; -0.4; -0.3; -0.2; -0.1|]
        |> Mat.of_array (Row 4) (Col 5)

let%test "shadow_submatrix" =
  let open Mat in
  let m2 = of_array (Row 2) (Col 2) [| -0.7;-0.6;-0.2;-0.1|] in

  match m2 with
  | Some m -> 
     m1
     >>= shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2) 
     >>| compare_float m
     |> opt_to_bool
  | None -> false

let%test "add" =
  let open Mat in
  let t () = 
    let* m2 = of_array (Row 2) (Col 2) [| 0.1; 0.2; 0.3; 0.4|] in
    let* res = of_array (Row 2) (Col 2) [| -0.6;-0.4;0.1;0.3|] in
    m1
    >>= shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2)
    >>= add m2
    >>| compare_float res
  in
  t () |> opt_to_bool
    
let%test "sub" =
  let open Mat in
  let m2 = of_array (Row 2) (Col 2) [| 0.1; 0.2; 0.3; 0.4|] |> Option.get in
  let res = of_array (Row 2) (Col 2) [| 0.8;0.8;0.5;0.5|] |> Option.get in

  m1
  >>= shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2) 
  >>= sub m2
  >>| compare_float res
  |> opt_to_bool

let%test "flatten" =
  let open Mat in
  let m4 = of_array (Row 1) (Col 2) [| -0.7; -0.6;|] |> Option.get in
  let m5 = of_array (Row 1) (Col 2) [| -0.2; -0.1;|] |> Option.get in
  let res = of_array (Row 2) (Col 1) [| m4; m5|]
            >>= flatten
            >>= reshape (Row 2) (Col 2) |> Option.get
  in

  m1
  >>= shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2) 
  >>| compare_float res
  |> opt_to_bool

let%test "convolve" =
  let open Mat in
  let test () =
    let* im = Array.init 9 (fun i -> float_of_int i +. 1.)
              |> of_array (Row 3) (Col 3) in
    let* kern = Array.init 4 (fun i -> float_of_int i +. 10.)
              |> of_array (Row 2) (Col 2) in
    convolve im ~stride:1 kern
  in

  match test () with
  | Some r ->
     print r; true
  | None -> false

let%test "mult" =
  let open Mat in
  let m2 = of_array (Row 2) (Col 1) [| 0.1;
                                       0.2; |] |> Option.get in
  let res = of_array (Row 2) (Col 1) [| -0.19;-0.04;|] |> Option.get in
  mult (shadow_submatrix (Row 2) (Col 3) (Row 2) (Col 2) (m1 |> Option.get) |> Option.get) m2
  >>| compare_float res 
  |> opt_to_bool

