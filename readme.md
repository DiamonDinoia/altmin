To run cern dataset with altmin using cpp implementation:

In artifacts folder run: python train_model --strategy altmin --cpp 1

Need to download the delphes TTbarFull train set

Also need libtorch downloaded locally atm but this is a priority to fix.

N.b:

Use from_blob to create a view of the python data in cpp. Then by doing in place operations on the tensor the changes made are also made to the underlying data. 
		
Todo:

1) Implement more layers (Have linear, ReLU and Sigmoid so far)
2) Time comparison 
3) Manually implement the autograd logic so can use eigen instead of torch 
4) At the moment code only works for double tensors, so expand by allowing the datatype of the tensors to be passed as a parameter 
5) Improve the adam and autograd units tests 
6) Add torch to the build system in cpp. At the moment I just use libtorch that was installed locally (I forgot about this and I'll have a look now)
7) Move manual altmin functions into for of altmin. This is just altmin but implementing the adam algorithm and weight updates manually as it made it easier to test against cpp
8) Move datasets folder outside of main package.
9)
10)
