# VFX_project2 -- Image Stitching

## Members
- B09902029 方佑心
- B09902048 王品文

## Execution
```bash=
python main.py [-h] [--root ROOT] [--result_path RESULT_PATH] [--focal_len FOCAL_LEN]
```

**Optional arguments**
- `--root`: The root of input images.
- `--result_path`: The path of the output panorama.
- `--focal_len`: The focal length of the input images.

## Dependency
matplotlib==3.8.4
numpy==1.26.4
opencv-python==4.9.0.80
pandas==2.2.2
scipy==1.13.0
tqdm==4.66.2
