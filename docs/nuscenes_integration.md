# nuScenes Testing Guide

This document explains how to run MCTrack on the nuScenes dataset when you have your
own LiDAR detections and RGB images. The detection files must be converted to the
BaseVersion format before running the tracker.

## Directory structure
Place the raw nuScenes data and detection results in the following locations:

```text
data/
└── nuScenes/
    ├── datasets/                # Original nuScenes data (maps, samples, sweeps, ...)
    └── detectors/
         └── <detector_name>/    # e.g. centerpoint or largekernel
             ├── val.json       # validation detections
             └── test.json      # test detections
```

After the files are prepared, convert them to the internal BaseVersion format:

```bash
python preprocess/convert2baseversion.py --dataset nuscenes
```

The converted files will be saved under `data/base_version/nuscenes/` and can be
used directly by `main.py`.

## Running tracking
Run the tracker and evaluation with:

```bash
python main.py --dataset nuscenes -e -p 1
```

The results will be written to `results/nuscenes/<date_time>/`.
