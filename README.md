# BayesFGM

**Paper**: Jiajing Niu, Andrew Brown. Bayesian regularization for functional graphical models. 

# Requirements
- R version 3.6.3 and up
- Python 3.6 and up
- TensorFlow 2.6.0
- Tensorflow-probability 0.14.1

# Basic Example

Run `python example.py` to excute the 10 node model by functional graphical horseshoe.

Optional arguments which may be of the most interest: 

- `p` and `N` is arguments for number of nodes and data size respectively.
- You can modify `prior` argument to choose different Bayesian models, i.e., "lasso", "horseshoe"
- `data_type` argument is for choosing regular dense data or irregular sparse data.
- `Net_num` argument is for choosing the true underlying network connection.
 

# Train your data

The example is simply using our own data generator. In order to evaluate on similar data as in the paper, you will need to compute the fpc scores and use funcions in Bayesmodel/Bayesmodel.py  

# Need help?

For questions or bug reports, please email jiajinn@g.clemson.edu (or ab7@clemson.edu). 
