# PATE Example

The scripts in the folder allow you to train a MNIST model using PATE diffrential privacy framework.
While running this example would give you an accurate implementation of a PATE implementation. An accurate analysis of DP guarantees is still an work in progress.

[Sorry, I wasn't a very good coder when I wrote this. But, not maintaining this anymore] 


<h2>Requirements:</h2>

<ul>
 <li>Pytorch</li>
 <li>PySyft</li>
</li>
</ul>

```
$ python Main.py
```

Scripts present
 * data: Consists of functions for loading datasets
 * Main: The file to be run for a complete PATE model
 * Model: PyTorch model definition. The same model is used for student and teacher.
 * Student: Class to handle student functionality such as training and making predictions
 * Teacher: Class to handle teacher functionality such as training and making noisy predictions. All the Teacher ensembles are handled in this Class
 * util: Functions


 This training loop is then executed for a given number of epochs.
 The performance on the test set of MNIST is shown after each epoch.
