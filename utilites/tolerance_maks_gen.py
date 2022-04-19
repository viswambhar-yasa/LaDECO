# -*- coding: utf-8 -*-
## This file contain function based on probabilty (tolerance) the segmentation operarion can be performed
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)


import numpy as np


def tolerance_predicted_mask(predicted_mask, tol=0.5):
    """

    Args:
        predicted_mask (_type_): Predicted mask
        tol (float, optional): upper limit of probabity based on which the values are penalized. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    predicted_mask[predicted_mask > tol] = 1
    predicted_mask[predicted_mask < tol] = 0
    predicted = np.squeeze(predicted_mask)
    final_mask = np.zeros((256, 256))
    # creating a single mask
    for i in range(4):
        temp = predicted[:, :, i]
        final_mask[temp == 1] += i
    return final_mask
