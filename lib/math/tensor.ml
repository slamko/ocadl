open Common
open Bigarray

module Vec = struct
  type shape = {
      dim1 : col;
    }

  let shape_size v = col v.dim1

  let get_shape vec =
    { dim1 = Col ( Array1.dim vec ) }

end

module Mat = struct
  type shape = {
      dim1 : row;
      dim2 : col;
    }

  let shape_size m = row m.dim1 * col m.dim2

  let get_shape mat =
    { dim1 = Row (Array2.dim1 mat); dim2 = Col (Array2.dim2 mat) }

end

module Mat3 = struct
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

end

