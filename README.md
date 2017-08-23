# PKB usage instruction
This webpage provides an instruction for using the PKB (Pathway-based Kernel Boosting) model. PKB is designed to perform classification analysis with gene expression data. It incorporates gene pathway information as prior knowledge, and performs selection of informative pathways at the same time as building the predictive model. 

**Software preparation**:   
The program is written in **Python3**. Please install the following Python packages before running the program: 

- pandas
- numpy
- sharedmem
- scipy
- multiprocessing
- yaml
- matplotlib

## Data preparation
### Clinical outcome input
### Gene expression input
### Pathway input
### PKB configuration file
Here is an example configuration file for applying PKB to our example dataset:

```python
# folders
input_folder: ./example
output_folder: example_output

# input files
predictor: predictor.txt  
response: response.txt    
predictor_set: predictor_sets.txt 

# model parameters
maxiter: 500
random_subset: 1
learning_rate: 0.02
Lambda:   
kernel: rbf
method: L1
test_ratio: 0.33 
```

The parameters are interpreted as following:

- `input_folder`: the folder where you keep the input data (path relative to your current folder)
- `output_folder`: the folder where you want to PKB to keep the output figures and data (path relative to `input_folder`)
- `predictor`: input gene expression file (path relative to `input_folder`)
- `response`: input clinical outcome file (path relative to `input_folder`)
- `predictor_set`: input pathway file (path relative to `input_folder`)
- `maxiter`: number of maximum boosting iterations
- `random_subset`: a value in `[0,1]`. Each time choose a random subset of samples to train the model. Choose a value `< 1` to speed up the boosting process when the dataset is large 
- `learning_rate`: the learning rate parameter $\nu$ 
- `Lambda`: the penalty parameter. If left blank, PKB will use an auto-determine algorithm to choose one.
- `kernel`: the kernel function. Currently we support, radial basis function(`rbf`) and polynomial kernel with $k$ degrees (`poly2`,`poly3`,etc)
- `method`: `L1` for $L_1$ penalty, `L2` for $L_2$ penalty
- `test_ratio`: the proportion of data to be used as testing dataset


## Running PKB
Follow the steps below in order to run PKB on your own computer (we use our toy dataset as example):

1. clone this git repository :

	```bash
	git clone https://github.com/zengliX/PKB PKB
	cd PKB
	```
2. prepare datasets and configuration files following the format given in the previous section

3. implement PKB: 

	```
	python PKB.py ./example/config_file.txt
	```

The outputs will be saved in the `output_folder` as you specified in the configuration file.
## Results interpretation

## Contact 
Please feel free to contact <li.zeng@yale.edu> if you have any questions.