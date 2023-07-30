open Common

type col =
  | Col of int [@@deriving show]

type row =
  | Row of int [@@deriving show]

module Mat = struct 
 type 'a t = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    }
   [@@deriving show]

type size =
  | Empty
  | Size of int

end

type shape = {
    dim1 : row;
    dim2 : col;
    dim3 : int;
  } [@@deriving show]

open Mat

exception InvalidIndex

let get_row (Row row) = row

let get_col (Col col) = col

let shape_size shape =
  get_row shape.dim1
  |> ( * ) @@ get_col shape.dim2 
  |> ( * ) shape.dim3

let size mat = get_row mat.rows |> ( * ) @@ get_col mat.cols

let get_size mat =
  match size mat with
  | 0 -> Empty
  | size -> Size size

let make_shape3d dim1 dim2 dim3 = 
  if get_row dim1 < 0 || get_col dim2 < 0 || dim3 < 0
  then failwith "Invalid shape."
  else { dim1;
         dim2;
         dim3;
       }

let make_shape rows cols = make_shape3d rows cols 1

let get_shape mat =
  make_shape mat.rows mat.cols

let get_shape3d mat =
  make_shape3d 

let shape_match mat1 mat2 =
  let shape = get_shape mat2 in
  match get_shape mat1 |> compare shape with
  | 0 -> Ok shape
  | _ -> Error "Matrix shapes do not match."

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

let make (Row rows) (Col cols) init_val =
  Array.make (rows * cols) init_val
  |> of_array (Row rows) (Col cols)

let of_shape init_val shape =
  make shape.dim1 shape.dim2 init_val

let empty_shape () =
  { dim1 = Row 0;
    dim2 = Col 0;
    dim3 = 1;
  }

let zero_of_shape shape =
  of_shape 0. shape

let qcopy mat =
  of_array mat.rows mat.cols (Array.copy mat.matrix)

let create (Row rows) (Col cols) finit =
  Array.init (rows * cols) (fun i ->
      finit (Row (i / cols)) (Col (i mod cols)))
  |> of_array (Row rows) (Col cols)

let empty () = of_array (Row 0) (Col 0) [| |]

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

let get_first mat = get (Row 0) (Col 0) mat

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

let iter proc mat =
  for r = 0 to get_row mat.rows - 1
  do for c = 0 to get_col mat.cols - 1
     do proc @@ get (Row r) (Col c) mat;
     done
  done

let random row col =
  create row col (fun _ _ -> (Random.float 2. -. 1.))

let random_of_shape shape =
  random shape.dim1 shape.dim2

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

let iter2 proc mat1 mat2 =
  let@ _ = shape_match mat1 mat2 in
  for r = 0 to get_row mat1.rows - 1
  do for c = 0 to get_col mat1.cols - 1
     do
       get (Row r) (Col c) mat2
       |> proc @@ get (Row r) (Col c) mat1 ;
     done
  done;
  ()

let iteri2 proc mat1 mat2 =
  let@ _ = shape_match mat1 mat2 in
  for r = 0 to get_row mat1.rows - 1
  do for c = 0 to get_col mat1.cols - 1
     do get (Row r) (Col c) mat2
        |> proc (Row r) (Col c) @@ get (Row r) (Col c) mat1;
     done
  done;
  ()

let set_raw row col mat value =
  set (Row row) (Col col) mat value

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

let reshape_of_shape mat shape  =
  reshape shape.dim1 shape.dim2 mat

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

let reshape3d base mat =
  let index = ref 0 in
  map (fun inner ->
      let subm = submatrix
                   (Row 0) (Col !index)
                   (Row 1) (Col (size inner)) mat in
      index := !index + size inner ;
      subm
    ) base

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

let fold_right proc mat init =
  let acc = ref init in
  iter (fun el -> acc := proc el !acc) mat;
  !acc

let fold_left proc init mat =
  let acc = ref init in
  iter (fun el -> acc := proc !acc el) mat;
  !acc

let fold_left2 proc init mat1 mat2 =
  let@ _ = shape_match mat1 mat2 in
  let acc = ref init in
  let _ = iter2 (fun val1 val2-> acc := proc !acc val1 val2) mat1 mat2 in
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
  | [| |] -> empty ()
  | mat_arr ->
     mat_arr
     |> of_array (Row 1) (Col (Array.length mat_arr))
     |> flatten

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

let mult mat1 mat2 =
  if get_col mat1.cols <> get_row mat2.rows
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

let convolve mat ~padding ~stride out_shape kernel =
  (* let kern_arr = kernel |> Mat.to_array in *)
  let kern_rows = dim1 kernel |> get_row in
  let kern_cols = dim2 kernel |> get_col in

  let mat_rows = dim1 mat |> get_row in
  let mat_cols = dim2 mat |> get_col in
  
  let base = create
               (Row (mat_rows + (2 * padding)))
               (Col (mat_cols + (2 * padding)))
               (fun (Row r) (Col c) ->
                 if r >= padding && c >= padding
                    && r < mat_rows + padding
                    && c < mat_cols + padding
                 then get_raw (r - padding) (c - padding) mat
                 else 0.
               ) in

  let base_rows = dim1 base |> get_row in
  let base_cols = dim2 base |> get_col in
  let res_mat = zero_of_shape out_shape in
  
  let rec convolve_rec kernel mat r c res_mat =
    if r + kern_rows > base_rows 
    then res_mat
    else
      if c + kern_cols > base_cols
      then convolve_rec kernel mat (r + stride) 0 res_mat
      else
        (* let a = 4 in *)
        (* Printf.eprintf "r: %d; c: %d\n" r c ; *)
        let submat = shadow_submatrix (Row r) (Col c)
                       kernel.rows kernel.cols base in
        let@ _ = shape_match kernel submat in
        let conv = fold_left2
                     (fun acc val1 val2 -> acc +. (val1 *. val2))
                     0. submat kernel in
        set_raw (r / stride) (c / stride) res_mat conv;
        convolve_rec kernel mat r (c + stride) res_mat
    in

    convolve_rec kernel mat 0 0 res_mat

