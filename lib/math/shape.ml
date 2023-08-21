open Common
open Bigarray
open Alias
open Tensor

type _ shape =
  | ShapeMatVec : Mat3.shape -> mat3 tensor shape
  | ShapeMat    : Mat.shape -> mat tensor shape
  | ShapeVec    : Vec.shape -> vec tensor shape

let shape_size : type a. a shape -> int =
  function 
  | ShapeVec v -> Vec.shape_size v
  | ShapeMat m -> Mat.shape_size m
  | ShapeMatVec m3 -> Mat3.shape_size m3

let get_shape : type a. a tensor -> a tensor shape =
  fun tensor ->
  match tensor with
  | Tensor1 x -> ShapeVec (Vec.get_shape x)
  | Tensor2 x -> ShapeMat (Mat.get_shape x)
  | Tensor3 mat3 -> ShapeMatVec (Mat3.get_shape mat3)

let shape_eq : type a. a shape -> a shape -> bool =
  fun shape1 shape2 ->
  match shape1, shape2 with
  | ShapeMatVec m1, ShapeMatVec m2 ->
     (if compare m1 m2 <> 0 
      then false
      else true)
  | ShapeVec v1, ShapeVec v2 ->
     (if compare v1 v2 <> 0 
      then false
      else true)
  | ShapeMat m1, ShapeMat m2 ->
     (if compare m1 m2 <> 0 
      then false
      else true) 

let make_shape_vec vec_shape =
  ShapeVec vec_shape

let make_shape_mat mat_shape =
  ShapeMat mat_shape

let make_shape_mat_vec mat3_shape =
  ShapeMatVec mat3_shape

let shape_match mat1 mat2 =
  let shape = get_shape mat2 in
  match get_shape mat1 |> compare shape with
  | 0 -> ()
  | _ -> failwith "Matrix shapes do not match."
 
