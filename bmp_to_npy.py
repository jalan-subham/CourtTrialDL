import glob 
from rich import print
import matplotlib.pyplot as plt
import numpy as np
import tqdm 

folder_paths = glob.glob("CourtTrial/Clips/Identities/*/*/*/*/*aligned/")

# first_folder = folder_paths[0]

# print(sorted(glob.glob(first_folder + "/*.bmp")))

for folder in tqdm.tqdm(folder_paths):
    bmps = sorted(glob.glob(folder + "/*.bmp"))
    output_arr = []
    for bmp in bmps:
        bmp_arr = plt.imread(bmp)
        output_arr.append(bmp_arr)
    output_arr = np.stack(output_arr)
    # output_arr = np.pad(output_arr, ((0, 2443 - output_arr.shape[0]), (0, 0), (0, 0), (0, 0)))
    print(output_arr.shape)
    # output_arr = output_arr / 255.0
    np.save(folder + "/video.npy", output_arr)

# npy_files = glob.glob("CourtTrial/Clips/Identities/*/*/*/*/*aligned/video.npy")
# for file in npy_files:
#     video = np.load(file)
#     video = np.pad(video, ((0, 2443 - video.shape[0]), (0, 0), (0, 0), (0, 0)))
#     video = video / 255.0
#     np.save(file, video)
    