open Common

type col =
  | Col of int [@@deriving show]

type row =
  | Row of int [@@deriving show]


 type 'a t = {
      matrix : 'a array;
      rows : row;
      cols : col;

      start_row : row;
      start_col : col;
      stride : int;
    }
   [@@deriving show]

  exception InvalidIndex

  type shape = {
      size : int;
      dim1 : row;
      dim2 : col;
    }

  type size =
    | Empty
    | Size of int

  let get_row (Row row) = row

  let get_col (Col col) = col

  let size mat = get_row mat.rows |> ( * ) @@ get_col mat.cols

  let get_size mat =
    match size mat with
    | 0 -> Empty
    | size -> Size size

  let get_shape mat =
    { size = size mat;
      dim1 = mat.rows;
      dim2 = mat.cols;
    }

  let shape_match mat1 mat2 =
    let shape = get_shape mat2 in
    match get_shape mat1 |> compare shape with
    | 0 -> Ok shape
    | _ -> Error "Matrix shapes do not match."

  let of_array rows cols matrix =
    if (get_row rows * get_col cols) > (matrix |> Array.length)
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

  let create (Row rows) (Col cols) finit =
    Array.init (rows * cols) (fun i ->
        finit (Row (i / rows)) (Col (i mod cols)))
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
                       (mat1 |> get (Row 0) (Col 0))
                       (mat2 |> get (Row 0) (Col 0))
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
       let res_mat = proc (Row 0) (Col 0) (mat |> get_first)
                     |> make mat.rows mat.cols in

       iteri (fun r c value1 ->
           proc r c value1 |> set r c res_mat)
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
      || get_col start_col + get_col cols > get_col mat.cols
    then invalid_arg "Submatrix: Index out of bounds."
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
        if r = mat_rows - kern_rows + 1
        then res_mat
        else
        if c + kern_cols > mat_cols
        then convolve_rec kernel stride mat (r + stride) 0 res_mat
        else
          (* let a = 4 in *)
          (* Printf.eprintf "r: %d; c: %d\n" r c ; *)
          let submat = shadow_submatrix (Row r) (Col c)
                          kernel.rows kernel.cols mat in
          let@ _ = shape_match kernel submat in
          let conv = fold_left2
                        (fun acc val1 val2 -> acc +. (val1 *. val2))
                        0. submat kernel in
          set_raw (r / stride) (c / stride) res_mat conv;
          convolve_rec kernel stride mat r (c + stride) res_mat
    in

    convolve_rec kernel stride mat 0 0 res_mat
