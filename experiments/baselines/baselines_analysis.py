import torch
import os 

# Iterating through each baseline's performance metrics
for filename in os.listdir():
    if filename == 'baselines_analysis.py': continue
    model_dict = torch.load(filename, map_location='cpu')
    # Retrieving the test performance of each baseline
    test_stats = model_dict['test_stats']
    print(filename)
    print(test_stats)