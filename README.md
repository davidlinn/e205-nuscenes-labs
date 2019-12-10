# E205 Nuscenes Labs

1) Download "Mini" Nuscenes dataset (3.88 GB) at https://www.nuscenes.org/download, noting that the parent directory of "v1.0-mini" will be called the ```dataroot```.
2) Clone https://github.com/nutonomy/nuscenes-devkit
3) Edit ```dataroot``` parameter in ```nuscenes-devkit/python-sdk/nuscenes/nuscenes.py``` and ```e205-nuscenes-labs/e205_devkit.py``` in accordance with #1

```
cd e205-nuscenes-labs
ipython3
 [ run e205_devkit.py
 [ testFilter() # lots of blocking matplotlib windows, must close terminal to quit
```
