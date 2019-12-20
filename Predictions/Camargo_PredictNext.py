import Camargo.predict_next as predict

output_folder = "output_run3/BPIC12_20000000_2_1_0_0_1/shared_cat"
model_file = "model_rd_100_Nadam_04-2.74.h5"

predict.predict_next(output_folder, model_file, False)


