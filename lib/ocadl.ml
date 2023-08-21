
let usage_msg = "ocadl -l <train_data_file> -s <save_file> -i <epoch_num> -b <batch_size>"

let epoch_num = ref 11
let batch_size = ref 1
let learning_rate = ref 0.01
let train_data_file = ref ""
let save_file = ref ""
let arch = ref []

let anon_fun layer =
  arch := (int_of_string layer)::!arch

let speclist =
  [
    ("-i", Arg.Set_int epoch_num, "Epochs count") ;
    ("-b", Arg.Set_int batch_size, "Batch size") ;
    ("-r", Arg.Set_float learning_rate, "Learning rate") ;
    ("-l", Arg.Set_string train_data_file, "Training data file name") ;
    ("-s", Arg.Set_string save_file, "Json file to dump the NN state")
  ]

let () =
  Unix.time () |> int_of_float |> Random.init ;
  Arg.parse speclist anon_fun usage_msg ;

    begin match
      Test.test 
         !train_data_file
         !save_file
         !epoch_num
         !learning_rate
         !batch_size
    with
    | Ok ok -> ok
    | Error err -> Printf.eprintf "error: %s\n" err end
