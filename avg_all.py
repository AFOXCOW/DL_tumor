import os
from tqdm import tqdm
for i in tqdm(range(10)):
	command = "python avg_norm_ggcam.py --folder_no "+str(i)
	os.system(command)