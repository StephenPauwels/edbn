import process_results as pr

results = pr.read_result("split_method.txt")

for d in pr.DATA:
    data_result = results[results.data == d]
    pr.plot_acc_dots(data_result, "split_method", title=d)
