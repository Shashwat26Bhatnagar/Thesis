"""
Repeat the same test file with different random seeds
"""
import os

seed_list = range(1, 51)
file_name = "test_mcpilco_cartpole.py"


for seed in seed_list:
    str_command = "python " + file_name + " -seed " + str(seed)
    print("\n##########\nstr_command: " + str_command + "\n##########")
    os.system(str_command)
