""" This file contains constant data structures that only need to be defined once. """

# land cover classes for all 4 schemes avaliable in the dataset
# created from Table 2 of the original paper 

igbp_classes = {
    1: 'Evergreen Needleleaf Forests',
    2: 'Evergreen Broadleaf Forests',
    3: 'Deciduous Needleleaf Forests',
    4: 'Deciduous Broadleaf Forests',
    5: 'Mixed Forests',
    6: 'Closed (Dense) Shrublands',
    7: 'Open (Sparse) Shrublands',
    8: 'Woody Savannas',
    9: 'Savannas',
    10: 'Grasslands',
    11: 'Permanent Wetlands',
    12: 'Croplands',
    13: 'Urban and Built-Up Lands',
    14: 'Cropland/Natural Vegetation Mosaics',
    15: 'Permanent Snow and Ice',
    16: 'Barren',
    17: 'Water Bodies'
}

lccs_lc_classes = {
    1: 'Barren',
    2: 'Permanent Snow and Ice',
    3: 'Water Bodies',
    11: 'Evergreen Needleleaf Forests',
    12: 'Evergreen Broadleaf Forests',
    13: 'Deciduous Needleleaf Forests',
    14: 'Deciduous Broadleaf Forests',
    15: 'Mixed Broadleaf/Needleleaf Forests',
    16: 'Mixed Broadleaf Evergreen/Deciduous Forests',
    21: 'Open Forests',
    22: 'Sparse Forests',
    31: 'Dense Herbaceous',
    32: 'Sparse Herbaceous',
    41: 'Closed (Dense) Shrublands',
    42: 'Shrubland/Grassland Mosaics',
    43: 'Open (Sparse) Shrublands'
}

lccs_lu_classes = {
    1: 'Barren',
    2: 'Permanent Snow and Ice',
    3: 'Water Bodies',
    9: 'Urban and Built-Up Lands',
    10: 'Dense Forests',
    20: 'Open Forests',
    25: 'Forest/Cropland Mosaics',
    30: 'Natural Herbaceous',
    35: 'Natural Herbaceous/Croplands Mosaics',
    36: 'Herbaceous Croplands',
    40: 'Shrublands'
}

lccs_sh_classes = {
    1: 'Barren',
    2: 'Permanent Snow and Ice',
    3: 'Water Bodies',
    10: 'Dense Forests',
    20: 'Open Forests',
    27: 'Woody Wetlands',
    30: 'Grasslands',
    40: 'Shrublands',
    50: 'Herbaceous Wetlands',
    51: 'Tundra'
}
