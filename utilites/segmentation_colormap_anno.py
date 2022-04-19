# -*- coding: utf-8 -*-
## This file contain function to plot segmentation mask with annotation 
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def segmentation_colormap_anno(pre_anno, heatmap='RdYlBu_r', file_name='seg_mask', disp=False):
    """
    To create segmentation map with annotation 

    Args:
        pre_anno (_type_): predicted annotation 
        heatmap (str, optional): colurmap type. Defaults to 'RdYlBu_r'.
        file_name (str, optional): file name to save the heat map. Defaults to 'seg_mask'.
        disp (bool, optional): parameter to plot the segmentation mask. Defaults to False.
    """
    plt.figure(figsize=(5,5))
    cmap = sns.mpl_palette("Set2", 4)
    # using seaborn heatmap to generate plot
    c = sns.heatmap(data=pre_anno, cmap=cmap, cbar=False)
    plt.axis('off')
    # labelling feature based on the value 
    legend_handles = [Patch(color=cmap[0], label='Coating'),  # red
                      Patch(color=cmap[1], label='Substrate'),
                      Patch(color=cmap[2], label='Damaged Substrate'),
                      Patch(color=cmap[3], label='Thermal band')]  # green
    plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='upper center', fontsize=12,
               handlelength=.8)
    fig = c.get_figure()
    # plotting image
    plt.imshow(pre_anno, cmap='RdYlBu_r')
    plt.title("Predicted Mask")
    if disp:
        file_path = "Documents/" + file_name + '.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
    pass
