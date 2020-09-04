<p align="center">
  <img width="60%" src="/img/d2l-pytorch.png" />
</p>

-----------------------------------------------------------------------------------------------------------

This project is adapted from the original [Dive Into Deep Learning](https://d2l.ai) book by Aston Zhang, Zachary C. Lipton, Mu Li, Alex J. Smola and all the community contributors. GitHub of the original book: [https://github.com/d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en). We have made an effort to modify the book and convert the MXnet code snippets into PyTorch.

Note: Some ipynb notebooks may not be rendered perfectly in Github. We suggest `cloning` the repo or using [nbviewer](https://nbviewer.jupyter.org/) to view the notebooks.

## Installation
Many of you will not have Python 3.6 already installed on your computers. Conda is an easy way to manage many different environments, each with its own Python versions and dependencies. This allows us to avoid conflicts between our preferred Python version and that of other classes. We’ll walk through how to set up and use a conda environment.

Prerequisite: Anaconda. Many of you will have it installed from classes such as EE 16A; if you don’t, install it through the link.
### Creating a Conda Environment
  ```conda create --name <env-name> python=3.6```

### Entering the Environment
  ```conda activate <env-name>```

### Setting the Environment
  ```pip install -r requirements.txt```


## Chapters

  * **Ch02 Installation**
    * [Installation](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch02_Installation/INSTALL.md)

  * **Ch03 Introduction**
    * [Introduction](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch03_Introduction/Introduction.ipynb)

  * **Ch04 The Preliminaries: A Crashcourse**
    * 4.1 [Data Manipulation](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch04_The_Preliminaries_A_Crashcourse/Data_Manipulation.ipynb)
    * 4.2 [Linear Algebra](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch04_The_Preliminaries_A_Crashcourse/Linear_Algebra.ipynb)
    * 4.3 [Automatic Differentiation](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch04_The_Preliminaries_A_Crashcourse/Automatic_Differentiation.ipynb)
    * 4.4 [Probability and Statistics](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch04_The_Preliminaries_A_Crashcourse/Probability_and_Statistics.ipynb)
    * 4.5 [Naive Bayes Classification](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch04_The_Preliminaries_A_Crashcourse/Naive_Bayes_Classification.ipynb)
    * 4.6 [Documentation](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch04_The_Preliminaries_A_Crashcourse/Documentation.ipynb)
    
  * **Ch05 Linear Neural Networks**
    * 5.1 [Linear Regression](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch05_Linear_Neural_Networks/Linear_Regression.ipynb)
    * 5.2 [Linear Regression Implementation from Scratch](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch05_Linear_Neural_Networks/Linear_Regression_Implementation_from_Scratch.ipynb)
    * 5.3 [Concise Implementation of Linear Regression](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch05_Linear_Neural_Networks/Concise_Implementation_of_Linear_Regression.ipynb)
    * 5.4 [Softmax Regression](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch05_Linear_Neural_Networks/Softmax_Regression.ipynb)
    * 5.5 [Image Classification Data (Fashion-MNIST)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch05_Linear_Neural_Networks/Image_Classification_Data(Fashion-MNIST).ipynb)
    * 5.6 [Implementation of Softmax Regression from Scratch](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch05_Linear_Neural_Networks/Implementation_of_Softmax_Regression_from_Scratch.ipynb)
    * 5.7 [Concise Implementation of Softmax Regression](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch05_Linear_Neural_Networks/Concise_Implementation_of_Softmax_Regression.ipynb)

  * **Ch06 Multilayer Perceptrons**
    * 6.1 [Multilayer Perceptron](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Multilayer_Perceptron.ipynb)
    * 6.2 [Implementation of Multilayer Perceptron from Scratch](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Implementation_of_Multilayer_Perceptron_from_Scratch.ipynb)
    * 6.3 [Concise Implementation of Multilayer Perceptron](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Concise_Implementation_of_Multilayer_Perceptron.ipynb)
    * 6.4 [Model Selection Underfitting and Overfitting](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Model_Selection_Underfitting_and_Overfitting.ipynb)
    * 6.5 [Weight Decay](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Weight_Decay.ipynb)
    * 6.6 [Dropout](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Dropout.ipynb)
    * 6.7 [Forward Propagation Backward Propagation and Computational Graphs](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Forward_Propagation_Backward_Propagation_and_Computational_Graphs.ipynb)
    * 6.8 [Numerical Stability and Initialization](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Numerical_Stability_and_Initialization.ipynb)
    * 6.9 [Considering the Environment](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Considering_The_Environment.ipynb)
    * 6.10 [Predicting House Prices on Kaggle](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch06_Multilayer_Perceptrons/Predicting_House_Prices_on_Kaggle.ipynb)

  * **Ch07 Deep Learning Computation**
    * 7.1 [Layers and Blocks](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch07_Deep_Learning_Computation/Layers_and_Blocks.ipynb)
    * 7.2 [Parameter Management](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch07_Deep_Learning_Computation/Parameter_Management.ipynb)
    * 7.3 [Deferred Initialization](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch07_Deep_Learning_Computation/Deferred_Initialization.ipynb)
    * 7.4 [Custom Layers](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch07_Deep_Learning_Computation/Custom_Layers.ipynb)
    * 7.5 [File I/O](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch07_Deep_Learning_Computation/File_I_O.ipynb)
    * 7.6 [GPUs](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch07_Deep_Learning_Computation/GPUs.ipynb)

  * **Ch08 Convolutional Neural Networks**
    * 8.1 [From Dense Layers to Convolutions](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch08_Convolutional_Neural_Networks/From_Dense_Layers_to_Convolutions.ipynb)
    * 8.2 [Convolutions for Images](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch08_Convolutional_Neural_Networks/Convolutions_For_Images.ipynb)
    * 8.3 [Padding and Stride](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch08_Convolutional_Neural_Networks/Padding_and_Stride.ipynb)
    * 8.4 [Multiple Input and Output Channels](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch08_Convolutional_Neural_Networks/Multiple_Input_and_Output_Channels.ipynb)
    * 8.5 [Pooling](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch08_Convolutional_Neural_Networks/Pooling.ipynb)
    * 8.6 [Convolutional Neural Networks (LeNet)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch08_Convolutional_Neural_Networks/Convolutional_Neural_Networks(LeNet).ipynb)

  * **Ch09 Modern Convolutional Networks**
    * 9.1 [Deep Convolutional Neural Networks (AlexNet)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/AlexNet.ipynb) 
    * 9.2 [Networks Using Blocks (VGG)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/VGG.ipynb)
    * 9.3 [Network in Network (NiN)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Network_in_Network(NiN).ipynb) 
    * 9.4 [Networks with Parallel Concatenations (GoogLeNet)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Networks_with_Parallel_Concatenations_(GoogLeNet).ipynb) 
    * 9.5 [Batch Normalization](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Batch_Normalization.ipynb)
    * 9.6 [Residual Networks (ResNet)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Residual_Networks_(ResNet).ipynb) 
    * 9.7 [Densely Connected Networks (DenseNet)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Densely_Connected_Networks_(DenseNet).ipynb) 

  * **Ch10 Recurrent Neural Networks**
    * 10.1 [Sequence Models](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Sequence_Models.ipynb)
    * 10.2 [Language Models](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Language_Models.ipynb)
    * 10.3 [Recurrent Neural Networks](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Recurrent_Neural_Networks.ipynb)
    * 10.4 [Text Preprocessing](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Text_Preprocessing.ipynb)
    * 10.5 [Implementation of Recurrent Neural Networks from Scratch](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Implementation_of_Recurrent_Neural_Networks_from_Scratch.ipynb)
    * 10.6 [Concise Implementation of Recurrent Neural Networks](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Concise_Implementation_of_Recurrent_Neural_Networks.ipynb)
    * 10.7 [Backpropagation Through Time](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Backpropagation_Through_Time.ipynb)
    * 10.8 [Gated Recurrent Units (GRU)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Gated_Recurrent_Units.ipynb)
    * 10.9 [Long Short Term Memory (LSTM)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Long_Short_Term_Memory.ipynb)
    * 10.10 [Deep Recurrent Neural Networks](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Deep_Recurrent_Neural_Networks.ipynb)
    * 10.11 Bidirectional Recurrent Neural Networks
    * 10.12 [Machine Translation and DataSets](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Machine_Translation_and_Data_Sets.ipynb)
    * 10.13 [Encoder-Decoder Architecture](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Encoder-Decoder_Architecture.ipynb) 
    * 10.14 [Sequence to Sequence](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Sequence_to_Sequence.ipynb)
    * 10.15 [Beam Search](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Beam_Search.ipynb)

  * **Ch11 Attention Mechanism**
    * 11.1 [Attention Mechanism](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch11_Attention_Mechanism/Attention_Mechanism.ipynb)
    * 11.2 Sequence to Sequence with Attention Mechanism
    * 11.3 Transformer

  * **Ch12 Optimization Algorithms**
    * 12.1 [Optimization and Deep Learning](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch12_Optimization_Algorithms/Optimization_And_Deep_Learning.ipynb)
    * 12.2 [Convexity](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch12_Optimization_Algorithms/Convexity.ipynb)
    * 12.3 [Gradient Descent](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch12_Optimization_Algorithms/Gradient_Descent.ipynb)
    * 12.4 [Stochastic Gradient Descent](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch12_Optimization_Algorithms/Stochastic_Gradient_Descent.ipynb)
    * 12.5 [Mini-batch Stochastic Gradient Descent](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch12_Optimization_Algorithms/Mini-batch_Stochastic_Gradient_Descent.ipynb)
    * 12.6 [Momentum](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch12_Optimization_Algorithms/Momentum.ipynb)
    * 12.7 Adagrad
    * 12.8 [RMSProp](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch12_Optimization_Algorithms/RMSProp.ipynb)
    * 12.9 Adadelta
    * 12.10 Adam
  * **Ch14 Computer Vision**
    * 14.1 Image Augmentation
    * 14.2 Fine Tuning
    * 14.3 [Object Detection and Bounding Boxes](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch14_Computer_Vision/Object_Detection_and_Bounding_Boxes.ipynb)
    * 14.4 [Anchor Boxes](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch14_Computer_Vision/Anchor_Boxes.ipynb)
    * 14.5 [Multiscale Object Detection](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch14_Computer_Vision/Multiscale_Object_Detection.ipynb)
    * 14.6 [Object Detection Data Set (Pikachu)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch14_Computer_Vision/Object_Detection_Data_Set.ipynb)
    * 14.7 [Single Shot Multibox Detection (SSD)](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch14_Computer_Vision/Single_Shot_Multibox_Detection.ipynb)
    * 14.8 Region-based CNNs (R-CNNs)
    * 14.9 Semantic Segmentation and Data Sets
    * 14.10 Transposed Convolution
    * 14.11 Fully Convolutional Networks (FCN)
    * 14.12 [Neural Style Transfer](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch14_Computer_Vision/Neural_Style_Transfer.ipynb)
    * 14.13 Image Classification (CIFAR-10) on Kaggle
    * 14.14 Dog Breed Identification (ImageNet Dogs) on Kaggle

## Contributing

  * Please feel free to open a Pull Request to contribute a notebook in PyTorch for the rest of the chapters. Before starting     out with the notebook, open an issue with the name of the notebook in order to contribute for the same. We will assign         that issue to you (if no one has been assigned earlier).

  * Strictly follow the naming conventions for the IPython Notebooks and the subsections.

  * Also, if you think there's any section that requires more/better explanation, please use the issue tracker to 
    open an issue and let us know about the same. We'll get back as soon as possible.

  * Find some code that needs improvement and submit a pull request.

  * Find a reference that we missed and submit a pull request.

  * Try not to submit huge pull requests since this makes them hard to understand and incorporate. 
    Better send several smaller ones.


## Support 

If you like this repo and find it useful, please consider (★) starring it, so that it can reach a broader audience.

## References

[1] Original Book [Dive Into Deep Learning](https://d2l.ai) -> [Github Repo](https://github.com/d2l-ai/d2l-en)

[2] [Deep Learning - The Straight Dope](https://github.com/zackchase/mxnet-the-straight-dope)

[3] [PyTorch - MXNet Cheatsheet](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/cheatsheets/pytorch_gluon.md)


## Cite
If you use this work or code for your research please cite the original book with the following bibtex entry.
```
@book{zhang2020dive,
    title={Dive into Deep Learning},
    author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
    note={\url{https://d2l.ai}},
    year={2020}
}
```
