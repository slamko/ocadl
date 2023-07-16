open Lacaml.D.Mat

module DeepMat = sig
   
  val mat_print : mat -> unit
  
  let mat_add mat1 mat2 : mat mat -> mat
  
  let mat_sub mat1 mat2 : mat mat -> mat
  
  let mat_add_const cst mat : float mat -> mat
   
  let mat_div_scale cst mat : float mat -> mat
  
  let mat_scale cst mat : float mat -> mat
  
  let mat_row_to_array col mat : int mat -> float array
  
end
