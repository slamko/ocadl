module DeepMat = struct
  
  let mat_print (mat : mat)  =
    Format.printf
      "\
       @[<2>Matrix :\n\
       @\n\
       %a@]\n\
       @\n"
      Lacaml.Io.pp_fmat mat
  
  
  let mat_add mat1 mat2 =
    Mat.add mat1 mat2
  
  let mat_sub mat1 mat2 =
    Mat.sub mat1 mat2
  
  let mat_add_const cst mat =
    Mat.add_const cst mat
   
  let mat_div_scale cst mat =
    Mat.map (fun v -> v /. cst) mat
  
  let mat_scale cst mat =
    Mat.map (fun v -> v *. cst) mat
  
  let mat_row_to_array col mat =
    Mat.to_array mat |> arr_get col
  
end
