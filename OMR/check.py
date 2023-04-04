from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def find_indices(a,d=6):
    sorted_a = np.sort(a)  
    groups = []  

    current_group = [sorted_a[0]]

    for i in range(1, len(sorted_a)):
        less_count = 0
        for indices in current_group[1:]:
            if sorted_a[i] - indices <= d:
                less_count += 1
            else:
                break
        
        if less_count == len(current_group) - 1:
            current_group.append(sorted_a[i])
        else:
            groups.append(current_group)
            current_group = [sorted_a[i]]

    groups.append(current_group)

    return groups


def find_start_points(img):
    arr = np.array(img)
    y = np.sum(arr,axis=1)
    y = np.argpartition(y, 220)[:220]
    groups = find_indices(y)
    indices = [x[0] for x in groups]

    return indices



