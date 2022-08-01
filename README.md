# NEST

**NEST: Neural Event Stack for Event-based Image Enhancement**, ECCV 2022

[Minggui Teng](https://tengminggui.cn/), Chu Zhou, Hanyue Lou, and [Boxin Shi](https://ci.idm.pku.edu.cn/).
___

## Dependency

```shell
pip install -r requirements.txt
```


## Dataset Folder

```
dataset
  └─test
      ├─blurred  
      │       000_00000001.png
      │       ......
      ├─lr  
      │       000_00000001.png
      │       ......      
      └─event
              000_00000001.npy
              ......                    
```

## Train

### Deblur
```shell
python main.py --train --mode deblur 
```
### SR
```shell
python main.py --train --mode sr


## Test

### Deblur
```shell
python main.py --mode deblur 
```
### SR
```shell
python main.py --mode sr
```

configurations can be changed in `utils/options.py`

## Examples of HFR video generation application
![examples1](./demo/example-v1.gif)


More reuslts are in the subfolder `results`.


## Contact
If you have any questions, please send an email to minggui_teng@pku.edu.cn