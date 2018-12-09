# DS222-project

This repository contains the code for DS222 project offered in 2018 by Dr. Partha Pratim Talukdar

The baseline used for this project is [Joint Representation Learning of Text and Knowledge for Knowledge Graph Completion](https://arxiv.org/abs/1611.04125)

Requirements for the project:
```
Tensorflow >= 1.5
python2
sklearn
```

To run the serial code, set the data paths and run using `python train.py`. Use `python test.py` to test the performance of the model.

Running the parallel version of the model requires us to execute on different workers as shown below:

```
python train_asynchro.py --job_name="ps" --task_index=0
python train_asynchro.py --job_name="worker" --task_index=0
python train_asynchro.py --job_name="worker" --task_index=1
```

The above three commands should be executed on three different machines. We are running one parameter server and two worker nodes with this command.
