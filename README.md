## GNRRW
# paper data and code
This is the code for the ICDM 2021 Paper: Graph Neighborhood Routing and Random Walk
for Session-based Recommendation. 

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup> 

## How to Use

You need to run the file `build_graph.py` first to generate the graph files.

Then you need to run the file `rwr.py` to obtain the item global distribution file.

At last, you can run the file `main_cls.py` to train the model.

Take Yoochoose 1/64 dataset as example:
```
python build_graph.py --dataset yoochoose1_64 --sample_num 12 --theta 2
python rwr.py --dataset yoochoose1_64 --anchor_num 40 --alpha 0.5
python main_cls.py --dataset yoochoose1_64 --n_factors 40
```

## Requirements

- Python3
- pytorch==1.4.0

## Citation
Please cite our paper if you use the code!
