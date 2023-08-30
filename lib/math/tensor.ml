open Common
open Bigarray

external cc_mat3_print : (float, float32_elt, c_layout) Array3.t ->
                        unit = "cc_mat_print"

external cc_mat_print : (float, float32_elt, c_layout) Array2.t ->
                        unit = "cc_mat_print"

external cc_vec_print : (float, float32_elt, c_layout) Array1.t ->
                        unit = "cc_mat_print"

external cc_mat3_free : (float, float32_elt, c_layout) Array3.t ->
                        unit = "cc_mat_free"

external cc_mat_free : (float, float32_elt, c_layout) Array2.t ->
                        unit = "cc_mat_free"

external cc_vec_free : (float, float32_elt, c_layout) Array1.t ->
                        unit = "cc_mat_free"

let randf () =
  Random.float 2.0 -. 1.0

module Vec = struct
  type shape = {
      dim1 : col;
    }

  type tensor = (float, float32_elt, c_layout) Array1.t
  
  type t =
    { matrix : tensor;
      shape : shape;
    }

  let shape_size v = col v.dim1

  let get_shape vec =
    { dim1 = Col ( Array1.dim vec.matrix ) }

  let make_shape cols =
    { dim1 = cols }

  let wrap big_arr =
    { matrix = big_arr;
      shape = { dim1 = Col (Array1.dim big_arr) }
    }

  let create big_arr =
    Gc.finalise cc_vec_free big_arr ;
    { matrix = big_arr;
      shape = { dim1 = Col (Array1.dim big_arr) }
    }

  let init (Col cols) f =
    { matrix = Array1.init Float32 C_layout cols f;
      shape = {dim1 = Col cols};
    }

  let make cols init_val =
    init cols (fun _ -> init_val)

  let zero cols = make cols 0.0

  let random cols = init cols (fun _ -> randf ())

  let print vec =
    cc_vec_print vec.matrix

  let of_list lst =
    lst
    |> Array.of_list
    |> Array1.of_array Float32 C_layout 

  let of_array arr =
    let matrix = Array1.of_array Float32 C_layout arr in
    { matrix ;
      shape = { dim1 = Col (Array1.dim matrix) }
    }

  let set (Col col) vec value =
    Array1.unsafe_set vec.matrix col value

end

module Mat = struct

  type shape = {
      dim1 : row;
      dim2 : col;
    }

  type tensor = (float, float32_elt, c_layout) Array2.t

  type t =
    { matrix : tensor;
      shape: shape;
    }

  let shape_size m = row m.dim1 * col m.dim2

  let get_shape mat =
    { dim1 = Row (Array2.dim1 mat.matrix);
      dim2 = Col (Array2.dim2 mat.matrix) }

  let make_shape dim1 dim2 =
    { dim1; dim2 }

  let wrap big_arr =
    { matrix = big_arr;
      shape = { dim1 = Row (Array2.dim1 big_arr);
                dim2 = Col (Array2.dim2 big_arr); }
    }

  let create big_arr =
    Gc.finalise cc_mat_free big_arr ;
    { matrix = big_arr;
      shape = { dim1 = Row (Array2.dim1 big_arr);
                dim2 = Col (Array2.dim2 big_arr); }
    }

  let init (Row rows) (Col cols) f =
    { matrix = Array2.init Float32 C_layout rows cols f;
      shape = { dim1 = Row rows; dim2 = Col cols }
    }

  let random rows cols = init rows cols (fun _ _ -> randf ())

  let print mat =
    cc_mat_print mat.matrix

  let of_list (Row rows) (Col cols) lst =
    if List.length lst < rows * cols || rows = 0 || cols = 0
    then failwith "Invalid list for matrix creation";

    let rec rec_of_list lst i cur_acc acc =
      match lst with
      | [] -> cur_acc::acc
      | h::t ->
         if i > cols
         then
           rec_of_list t 2 [h] (cur_acc::acc)
         else rec_of_list t (i + 1) (h::cur_acc) acc
    in

    let mat_lst =
      if rows = 1
      then [lst]
      else rec_of_list lst 1 [] []
    in

    let mat_arr = List.map (fun l -> l
                                     |> List.map (fun x -> x /. 255.0)
                                     |> List.rev
                                     |> Array.of_list)

                    mat_lst |> List.rev |> Array.of_list in

    (* Printf.printf "Size %d\n" @@ Array.length mat_arr.(27) ; *)
    { matrix = Array2.of_array Float32 C_layout mat_arr ;
      shape = {dim1 = Row rows; dim2 = Col cols; }
    }

  let make rows cols init_val =
    init rows cols (fun  _ _ -> init_val)

  let zero rows cols = make rows cols 0.0

end

module Mat3 = struct
  type shape = {
      dim1 : row;
      dim2 : col;
      dim3 : col;
    }

  type tensor = (float, float32_elt, c_layout) Array3.t

  type t =
    { matrix : tensor;
      shape: shape
    }

  let shape_size m = row m.dim1 * col m.dim2 * col m.dim3

  let make_shape dim1 dim2 dim3 =
    { dim1; dim2 ; dim3 }

  let get_shape mat =
    { dim1 = Row (Array3.dim1 mat.matrix);
      dim2 = Col (Array3.dim2 mat.matrix);
      dim3 = Col (Array3.dim3 mat.matrix)
    }

  let wrap matrix =
    { matrix;
      shape = { dim1 = Row (Array3.dim1 matrix);
                dim2 = Col (Array3.dim2 matrix);
                dim3 = Col (Array3.dim3 matrix) }
    }

  let create matrix =
    Gc.finalise cc_mat3_free matrix ;
    { matrix;
      shape = { dim1 = Row (Array3.dim1 matrix);
                dim2 = Col (Array3.dim2 matrix);
                dim3 = Col (Array3.dim3 matrix) }
    }

  let init (Row rows) (Col cols) (Col dim3) f =
    { matrix = Array3.init Float32 C_layout rows cols dim3 f;
      shape = { dim1 = Row rows; dim2 = Col cols; dim3 = Col dim3 }
    }

  let random rows cols dim3 =
    init rows cols dim3 (fun _ _ _ -> randf ())

  let make rows cols dim3 init_val =
    init rows cols dim3 (fun  _ _ _ -> init_val)

  let zero rows cols dim3 = make rows cols dim3 0.0

end

