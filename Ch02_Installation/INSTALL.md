# Installation
:label:`chapter_installation`

To get you up and running with hands-on experiences, we'll need you to set up with a Python environment, Jupyter's interactive notebooks, the relevant libraries, and the code needed to *run the book*.

## Obtaining Source Codes

The source code package containing all notebooks is available at https://github.com/dsgiitr/d2l-pytorch.git. Please clone it into your working folder.

```
git clone https://github.com/dsgiitr/d2l-pytorch.git
```

## Installing Running Environment

If you have both Python 3.5 or newer and pip installed, the easiest way is to install the running environment through pip. PyTorch is needed along with the basic dependencies like jupyter, numpy etc.

Before installing `pytorch`, please first check if you are able to access GPUs. If so, please go to [sec_gpu](#sec_gpu) for instructions to install a CUDA-supported `pytorch`. Otherwise, we can install the CPU version, which is still good enough for the first few chapters.  

```bash
pip install torch
```

Once the packages are installed, we now open the Jupyter notebook by

```bash
jupyter notebook
```

At this point open http://localhost:8888 (which usually opens automatically) in the browser, then you can view and run the code in each section of the book.

<h2 id="sec_gpu">GPU Support</h2>

:label:`sec_gpu`

By default PyTorch is installed without GPU support to ensure that it will run on any computer (including most laptops). Part of this book requires or recommends running with GPU. If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads), you should install a GPU-enabled PyTorch. 

If you have installed the CPU-only version, then remove it first by

```bash
pip uninstall torch
```

Then you need to find the CUDA version you installed. You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`.
Then visit [Pytorch Installation Guide](https://pytorch.org/) and follow the steps to install your PyTorch version compatible with the CUDA version already installed.
