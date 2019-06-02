# Zero-shot-Learning-via-Recurrent-Knowledge-Transfer
Codes for the paper 'Zero-shot Learning via Recurrent Knowledge Transfer'.


## Download the Data
You can download all data (image features, attribtues and word vectors of AwA, CUB and ImageNet) used in this paper 
from [google drive](https://drive.google.com/open?id=18YYOi5FxiBJ5TYLfOkzO3HGw_w-EveyY). 
Then, put the data and code in the same fold (root path of the project).


## Run the Code
Before running the code, you need to install two toolboxes, namely, Dimensionality Reduction toolbox (drtoolbox.tar.gz) and LeastR toolbox (SLEP_package_4.1.zip). 
First, download the two toolboxes from [google drive](https://drive.google.com/open?id=18YYOi5FxiBJ5TYLfOkzO3HGw_w-EveyY). Then, unzip drtoolbox.tar.gz and SLEP_package_4.1.zip. Finally, add the two toolboxes into your Matlab path with subfolders. 

For more support, you can visit [drtoolbox](https://lvdmaaten.github.io/drtoolbox/) and [LeastR](http://www.yelab.net/software/SLEP/). <br>

Now, you can run main.m. <br>
The cross validation is implemented in RecKT_CV.m. <br>
The algorithm (RecKT) is implemented in RecKT.m. <br>

## Citation
```
@inproceedings{zhao2019zero,
  title={Zero-Shot Learning Via Recurrent Knowledge Transfer},
  author={Zhao, Bo and Sun, Xinwei and Hong, Xiaopeng and Yao, Yuan and Wang, Yizhou},
  booktitle={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1308--1317},
  year={2019},
  organization={IEEE}
}
```
