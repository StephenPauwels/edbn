import Camargo.predict_next as predict

testing = []
#testing.append({"folder": "output_run3/BPIC12_20000000_2_1_0_0_1/shared_cat", "model": "model_rd_100_Nadam_04-2.74.h5"})
#testing.append({"folder": "output_run3/BPIC12_20000000_2_1_0_0_1/specialized", "model": "model_rd_100 Nadam_03-2.94.h5"})
#testing.append({"folder": "output_run3/BPIC12_20000000_2_1_0_1_1/shared_cat", "model": "model_rd_100_Nadam_01-1.12.h5"})
#testing.append({"folder": "output_run3/BPIC12_20000000_2_1_0_1_1/specialized", "model": "model_rd_100 Nadam_01-1.13.h5"})
#testing.append({"folder": "output_run3/BPIC12W_20000000_2_1_0_0_1/shared_cat", "model": "model_rd_100_Nadam_02-3.80.h5"})
#testing.append({"folder": "output_run3/BPIC12W_20000000_2_1_0_0_1/specialized", "model": "model_rd_100 Nadam_07-3.90.h5"})
#testing.append({"folder": "output_run3/BPIC12W_20000000_2_1_0_1_1/shared_cat", "model": "model_rd_100_Nadam_09-0.75.h5"})
#testing.append({"folder": "output_run3/BPIC12W_20000000_2_1_0_1_1/specialized", "model": "model_rd_100 Nadam_17-0.95.h5"})
#testing.append({"folder": "output_run3/BPIC15_20000000_2_1_0_0_1/shared_cat", "model": "model_rd_100_Nadam_05-8.13.h5"})
testing.append({"folder": "output_run3/BPIC15_20000000_2_1_0_0_1/specialized", "model": "model_rd_100 Nadam_04-8.00.h5"})
testing.append({"folder": "output_run3/BPIC15_20000000_2_1_0_1_1/shared_cat", "model": "model_rd_100_Nadam_01-7.52.h5"})
testing.append({"folder": "output_run3/BPIC15_20000000_2_1_0_1_1/specialized", "model": "model_rd_100 Nadam_04-7.31.h5"})
testing.append({"folder": "output_run3/HELPDESK_20000000_2_1_0_0_1/shared_cat", "model": "model_rd_100_Nadam_01-10.42.h5"})
testing.append({"folder": "output_run3/HELPDESK_20000000_2_1_0_0_1/specialized", "model": "model_rd_100 Nadam_01-10.63.h5"})
testing.append({"folder": "output_run3/HELPDESK_20000000_2_1_0_1_1/shared_cat","model": "model_rd_100_Nadam_50-0.32.h5"})
testing.append({"folder": "output_run3/HELPDESK_20000000_2_1_0_1_1/specialized","model": "model_rd_100 Nadam_62-0.55.h5"})



for test in testing:
    output_folder = test["folder"]
    model_file = test["model"]

    predict.predict_next(output_folder, model_file, False)


