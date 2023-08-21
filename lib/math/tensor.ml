open Common
open Bigarray

module Vec = struct
  type t =
    { matrix : (float, float32_elt, c_layout) Array1.t;
      dim : int;
    }

  type shape = {
      dim1 : col;
    }

  let shape_size v = col v.dim1

  let get_shape vec =
    { dim1 = Col ( Array1.dim vec ) }

  let make_shape cols =
    { dim1 = cols }

  let init (Col cols) f =
    Array1.init Float32 C_layout cols f

  let make cols init_val =
    init cols (fun _ -> init_val)

  let zero cols = make cols 0.0

  let of_list lst =
    lst
    |> Array.of_list
    |> Array1.of_array Float32 C_layout 

  let of_array arr =
    Array1.of_array Float32 C_layout arr

  let set (Col col) vec value =
    Array1.unsafe_set vec col value

end

module Mat = struct
  type t =
    { matrix : (float, float32_elt, c_layout) Array2.t;
      dim : int -> int;
    }

  type shape = {
      dim1 : row;
      dim2 : col;
    }

  let shape_size m = row m.dim1 * col m.dim2

  let get_shape mat =
    { dim1 = Row (Array2.dim1 mat); dim2 = Col (Array2.dim2 mat) }

  let make_shape dim1 dim2 =
    { dim1; dim2 }

  let init (Row rows) (Col cols) f =
    Array2.init Float32 C_layout rows cols f

  let make rows cols init_val =
    init rows cols (fun  _ _ -> init_val)

  let zero rows cols = make rows cols 0.0

end

module Mat3 = struct
  type t =
    { matrix : (float, float32_elt, c_layout) Array3.t;
      dim : int -> int -> int;
    }

  type shape = {
      dim1 : row;
      dim2 : col;
      dim3 : col;
    }

  let shape_size m = row m.dim1 * col m.dim2 * col m.dim3

  let get_shape mat =
    { dim1 = Row (Array3.dim1 mat);
      dim2 = Col (Array3.dim2 mat);
      dim3 = Col (Array3.dim3 mat)
    }

  let init (Row rows) (Col cols) (Col dim3) f =
    Array3.init Float32 C_layout rows cols dim3 f

  let make rows cols dim3 init_val =
    init rows cols dim3 (fun  _ _ _ -> init_val)

  let zero rows cols dim3 = make rows cols dim3 0.0

end

