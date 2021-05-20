""" A PyTorch wrapper for the DataLoader provided by authors of SEN12MS. """

import torch
import numpy as np

import utils.sen12ms_dataLoader as sen12ms

SUMMER = sen12ms.Seasons.SUMMER


class SEN12MSDataset(torch.utils.data.Dataset):
    """ PyTotch wrapper for the dataloader provided by the dataset authors. """

    def __init__(self, base_dir, season=SUMMER):
        self.base_dir = base_dir
        self.season = season
        self._dataset = sen12ms.SEN12MSDataset(base_dir=base_dir)

        # get a dictionary {scene_id: patch_ids} for the whole season
        season_ids = self._dataset.get_season_ids(season=season)

        # flatten it into a list of tuples unique for 256x256 patch
        # (scene_id, patch_id)
        self.patch_unique_ids = []

        for scene_id, patch_ids in season_ids.items():
            for patch_id in patch_ids:
                self.patch_unique_ids.append((scene_id, patch_id))

        self.lc_bands = sen12ms.LCBands.landuse
        self.s1_bands = sen12ms.S1Bands.ALL
        self.s2_bands = [
            sen12ms.S2Bands.B02,  # blue
            sen12ms.S2Bands.B03,  # green
            sen12ms.S2Bands.B04,  # red
            sen12ms.S2Bands.B08,  # near-infrared
            sen12ms.S2Bands.B05,  # red edge 1
            sen12ms.S2Bands.B06,  # red edge 2
            sen12ms.S2Bands.B07,  # red edge 3
            sen12ms.S2Bands.B08A,  # red edge 4
            sen12ms.S2Bands.B11,  # short-wave infrered 1
            sen12ms.S2Bands.B12,  # short-wave infrared 2
        ]

    def __len__(self):
        return len(self.patch_unique_ids)

    def __getitem__(self, idx):
        scene, patch = self.patch_unique_ids[idx]

        s1, _ = self._dataset.get_patch(self.season, scene, patch, self.s1_bands)
        s2, _ = self._dataset.get_patch(self.season, scene, patch, self.s2_bands)
        lc, _ = self._dataset.get_patch(self.season, scene, patch, self.lc_bands)
        
        # for some reason, some of the Sentinel-1 images (i.e., scene=146, patch=202)
        # have NaN values which completely screw everything up. took ages to find what's
        # wrong and why. the error PyTorch threw was at the accuracy computation:
        # ValueError: Probabilities in `preds` must sum up to 1 across the `C` dimension.
        # arrrgh damn you NaNs

        mean = np.nanmean(s1, axis=(1, 2), keepdims=True)
        np.nan_to_num(s1, copy=False, nan=mean)
        
        s1_image = s1 - mean
        s1_image /= s1.std(axis=(1, 2), keepdims=True)
        s1_image = torch.tensor(s1_image)

        s2_image = s2 - s2.mean(axis=(1, 2), keepdims=True)
        s2_image /= s2.std(axis=(1, 2), keepdims=True)
        s2_image = torch.tensor(s2_image)

        image = torch.empty(12, 256, 256, dtype=torch.float32)
        image[:2] = s1_image
        image[2:] = s2_image

        label = torch.zeros(256, 256, dtype=int)

        # combine classes 20 and 25; 30, 35, and 36
        # label[lc == 1] = 0
        label[lc == 2] = 1
        label[lc == 3] = 2
        label[lc == 9] = 3
        label[lc == 10] = 4
        label[lc == 20] = 5
        label[lc == 25] = 5
        label[lc == 30] = 6
        label[lc == 35] = 6
        label[lc == 36] = 6
        label[lc == 40] = 7

        return image, label


class SEN12MSDataset_64x64subpatches(torch.utils.data.Dataset):
    """ PyTotch wrapper for the dataloader provided by the dataset authors. """

    def __init__(self, base_dir, season=SUMMER):
        self.base_dir = base_dir
        self.season = season
        self._dataset = sen12ms.SEN12MSDataset(base_dir=base_dir)

        # get a dictionary {scene_id: patch_ids} for the whole season
        season_ids = self._dataset.get_season_ids(season=season)

        # flatten it into a list of tuples unique for each 64x64 patch
        # (scene_id, patch_id, subpatch_idx)
        self.subpatch_unique_ids = []

        for scene_id, patch_ids in season_ids.items():
            for patch_id in patch_ids:
                # there are 16 64x64 patches in 256x256 image
                for i in range(16):
                    self.subpatch_unique_ids.append((scene_id, patch_id, i))

        self.lc_bands = sen12ms.LCBands.landuse
        self.s1_bands = sen12ms.S1Bands.ALL
        self.s2_bands = [
            sen12ms.S2Bands.B02,  # blue
            sen12ms.S2Bands.B03,  # green
            sen12ms.S2Bands.B04,  # red
            sen12ms.S2Bands.B08,  # near-infrared
            sen12ms.S2Bands.B05,  # red edge 1
            sen12ms.S2Bands.B06,  # red edge 2
            sen12ms.S2Bands.B07,  # red edge 3
            sen12ms.S2Bands.B08A,  # red edge 4
            sen12ms.S2Bands.B11,  # short-wave infrered 1
            sen12ms.S2Bands.B12,  # short-wave infrared 2
        ]

        self.class_to_target_map = {
            0: 0,  # turns out, some subpatches [i.e. idx=54361] have mode NODATA, 0
            1: 0,
            2: 1,
            3: 2,
            9: 3,
            10: 4,
            20: 5,
            30: 6,
            40: 7,
        }

    def __len__(self):
        return len(self.subpatch_unique_ids)

    def __getitem__(self, idx):
        scene, patch, subpatch = self.subpatch_unique_ids[idx]

        s1, _ = self._dataset.get_patch(self.season, scene, patch, self.s1_bands)
        s2, _ = self._dataset.get_patch(self.season, scene, patch, self.s2_bands)
        lc, _ = self._dataset.get_patch(self.season, scene, patch, self.lc_bands)

        i = subpatch // 4  # row number of the 64x64 subpatch of the 256x256 image
        j = subpatch % 4  # column number of the 64x64 subpatch

        s1_subpatch = s1[:, i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64]
        s2_subpatch = s2[:, i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64]
        lc_subpatch = lc[:, i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64]

        # for some reason, some of the Sentinel-1 images (i.e., scene=146, patch=202)
        # have NaN values which completely screw everything up. took ages to find what's
        # wrong and why. the error PyTorch threw was at the accuracy computation:
        # ValueError: Probabilities in `preds` must sum up to 1 across the `C` dimension.
        # arrrgh damn you NaNs

        mean = np.nanmean(s1_subpatch, axis=(1, 2), keepdims=True)
        np.nan_to_num(s1_subpatch, copy=False, nan=mean)

        s1_image = s1_subpatch - mean
        s1_image /= s1_subpatch.std(axis=(1, 2), keepdims=True)
        s1_image = torch.tensor(s1_image)

        s2_image = s2_subpatch - s2_subpatch.mean(axis=(1, 2), keepdims=True)
        s2_image /= s2_subpatch.std(axis=(1, 2), keepdims=True)
        s2_image = torch.tensor(s2_image)

        image = torch.empty(12, 64, 64, dtype=torch.float32)
        image[:2] = s1_image
        image[2:] = s2_image

        # combine classes 20 and 25; 30, 35, and 36
        lc_subpatch[lc_subpatch == 25] = 20
        lc_subpatch[lc_subpatch == 35] = 30
        lc_subpatch[lc_subpatch == 36] = 30

        # use the most common value as the label
        values, counts = np.unique(lc_subpatch, return_counts=True)
        mode = values[np.argmax(counts)]
        label = self.class_to_target_map[mode]

        return image, torch.tensor(label, dtype=torch.long)
