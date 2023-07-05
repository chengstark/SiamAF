import subprocess

with open("output2(w_correction).txt", "w") as output:
    subprocess.call('python stat_significance_test.py', stdout=output, shell=True);