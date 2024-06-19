from glob import glob 
import os 

identities = glob("Real*/Clips/Identities/*/*/*.mp4")
rest = glob("Real*/Clips/Deceptive/*.mp4")
print(len(identities))

# # count = 0

# for i in identities:
#     if "lie" in i and "Deceptive" in i:
#         continue 
#     elif "truth" in i and i.split("/")[-2] == "Truthful":
#         continue 
#     else:
#         print("SOMETHING IS WRONG")
# count = 15
# for i in rest:
#     vid_name = i.split("/")[-1]
#     if "lie" in vid_name:
#         os.rename(i, f"Real-life_Deception_Detection_2016/Clips/Identities/{count}/Deceptive/{vid_name}")
#         count += 1
#     elif "truth" in vid_name:
#         os.rename(i, f"Real-life_Deception_Detection_2016/Clips/Identities/{count}/Truthful/{vid_name}")
#         count += 1
# while count < 53:
#     os.mkdir(f"Real-life_Deception_Detection_2016/Clips/Identities/{count}")
#     os.mkdir(f"Real-life_Deception_Detection_2016/Clips/Identities/{count}/Deceptive")
#     os.mkdir(f"Real-life_Deception_Detection_2016/Clips/Identities/{count}/Truthful")
#     count += 1

# dec = 0
# tru = 0

# for i in identities:
#     if "lie" in i:
#         dec += 1
#     elif "truth" in i:
#         tru += 1
# print(dec, tru)