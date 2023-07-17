open Lacaml.D
open Types
open Deepmath

let nn_print nn =
  print_string "\nNN print: \n" ;
  Printf.printf "Weights:\n" ;
  List.iter mat_print nn.wl ;
  Printf.printf "\nBiases:\n" ;
  List.iter mat_print nn.bl 

let list_print lst =
  List.iteri (fun i el -> Printf.printf "List element %d = %f\n" i el) lst

let list_split n lst =

  let rec split_rec n head tail =
    match n with
    | 0 -> (List.rev head, tail)
    | _ -> split_rec (n - 1) (List.hd tail :: head) (List.tl tail)
  in

  split_rec n [] lst

let list_to_mat n lst =
  (* let lst_arr = lst |> Array.of_list in *)
  (* let data_len = Array.length lst_arr in *)
  (* let nrows = data_len / n in *)
  (* let res_arr = Array.make_matrix nrows n 0. in *)

  (* Array.iteri *)
    (* (fun i num -> *)
      (* let row = i / nrows in *)
      (* let col = i mod nrows in *)
      (* res_arr.(row).(col) <- num ; ) lst_arr ; *)

  [lst] |> Mat.of_list

let list_parse_train_data in_cols pair_list =
  List.map (fun (res_list, data_list) ->
      ([res_list] |> Mat.of_list, data_list |> list_to_mat in_cols)) pair_list

let read_train_data fname res_len in_cols =
  let csv = Csv.load fname in
  csv
  |> List.map @@ List.map float_of_string 
  |> List.map @@ list_split res_len
  |> List.map
       (fun (res_list, data_list) ->
         let res_mat = Array.make_matrix 1 10 0. in
         let col = (List.hd res_list |> int_of_float) in
         res_mat.(0).(col) <- 1. ;
         (res_mat |> Mat.of_array,
          data_list |> list_to_mat in_cols))
  (* |> List.iter (fun (res, inp) -> mat_print inp) *)

   
let make_nn arch : nnet =

  let rec make_wl_rec arch nn_acc =
    match arch with
    | [] -> nn_acc
    | [_] -> nn_acc 
    | h::t ->
       make_wl_rec t (Mat.random (List.hd t) h :: nn_acc)
  in

  let rec make_bl_rec arch nn_acc =
    match arch with
    | [] -> nn_acc
    | [_] -> nn_acc 
    | h::t ->
       make_bl_rec t (Mat.random 1 h :: nn_acc)
  in

  let rev_arch = List.rev arch in
  {    
    wl = make_wl_rec rev_arch [] ;
    bl = make_bl_rec rev_arch [] ;
  }
   
let nn_apply proc nn1 nn2 =
  {
    wl = List.map2 proc nn1.wl nn2.wl;
    bl = List.map2 proc nn1.bl nn2.bl
  }

let nn_map proc nn =
  { wl = List.map proc nn.wl ;
    bl = List.map proc nn.bl ;
  }

let nn_zero nn =
  { wl = make_zero_mat_list nn.wl;
    bl = make_zero_mat_list nn.bl
  }

let get_data_input sample =
  snd sample

let get_data_out sample =
  fst sample
