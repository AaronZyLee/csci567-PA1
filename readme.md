# CSCI 567 Summer '18 — Machine Learning

## Programming Assignment 1

In this programming assignment, you will practice writing some of the learning algorithms that were discussed in class. In particular, you will implement a k-Nearest Neighbor (kNN) classifier, a Perceptron algorithm, and a Linear Regression; and then evaluate them. You will see how a machine learning pipeline is set up in general and how models are evaluated.

We have written a code execution pipeline for you in the form of jupyter notebook. You will have to complete the `Classes` and `Functions` as instructed in the jupyter notebook

### Setup

This assignment is to be developed in python3 (&ge; 3.5). Make sure you have the correct version installed. You will also need the following python packages :

-   numpy (1.13.1)
-   scipy (0.19.1)
-   scikit-learn (0.19.0)
-   matplotlib (2.0.2)
-   jupyter

There are multiple ways you can install python3. Below are some (in no order of preference)

-   You can use VirtualBox VM with all the setup by downloading from <http://bytes.usc.edu/files/cs103/VM/StudentVM_Spring2018.ova>. The password for user `cs567` is `developer`.
-   You can use anaconda distribution of python <https://www.anaconda.com/download/>.
-   You can use `virtualenv` to configure `python3` environment for this Programming assignment and more to come. You can also use `virtualenvwrapper` which contains a set of convenient scripts to help manage `virtualenv`s.
-   You can use native `python3` installation by downloading python3 from <https://www.python.org/> for Windows; however, you are strongly encouraged to use Anaconda instead. Linux and Mac may already have a `python3` installation.

To install packages, use `pip3`. (*N.B.* If you are inside `virtualenv` or `conda`, you might have to use `pip` instead. Make sure you are calling the correct `pip`; you can check this with `which pip` or `type -a pip`. Otherwise, the packages could end up somewhere other than your current environment.) The assignment has been tested with package versions mentioned above. If you face any difficulties with other versions, please bring them to our notice. We have listed the correct package versions in `requirements.txt`. To install all packages using that you can run `pip3 install --user -r requirements.txt`.

To work on the assignment once the setup is done.

-   Navigate to the repository folder and start `jupyter notebook` command.
-   Put the URL provided by jupyter in your browser. If you are on an older version of jupyter, you might have to manually type `localhost:8888` or `localhost:8000` in your browser, depending on the port dynamically chosen by jupyter.
-   Open `main.ipynb` from the listed files.
-   Read and follow the Instructions therein.

### Submission Instructions:

-   To submit just push the code to your private GitHub repository.
-   Do not change the file or folder names.
-   You can keep pushing the code to the github repository as much as you want. We will consider most recent commit before the date.
-   Make sure to submit the executed notebook with results (& plots); do **not** submit a cleared notebook.
-   Problem 1, 2, and 3 can be attempted in any order.
-   This programming assignment will **not** be graded.
<!-- -   You can submit the request for using one of the late days in the form posted on Piazza. -->

### Points

This programming assignment has three problems. All are within `main.ipynb` and are worth the following points:

-   Linear Regression (10 points)
-   k-Nearest Neighbor (10 points)
-   Perceptron (5 points)

### Further clarifications

- You are **not** allowed to use `scikit-learn` package to implement the required classes and functions.
- You can use Python standard built-in libraries.
- You can add your own code in *new* cells for verification, visualization, etc.
- Tuning lambda is part of solving a machine learning problem (data-dependent), but not part of algorithm implementation. In this homework, instead of solving one machine learning problem, you are only required to implement the algorithms (you don't need to worry about hyperparameter tuning for the homework grading). For picking the best lambda, you can do cross-validation.
- All three parts of the assignment follows the same API, so they all have the `train` function. What should be implemented in the `train` functions are specific to each algorithm.
- Technical support will mainly be provided for developing in the VM. If you decide to use your own set up for development, we can't promise that we will provide support for it.
- The polynomial features used in this assignment is a simplified version.
- For the purpose of this assignment, treat class 1 as the positive class and class 0 as the negative class.
- For kNN with even k, you can decide on a consistent way to break tie (equal number of the two labels).
- For data scaling, the global min and max values for each feature should be taken from the *current* training set. This includes validation set or not depending on the current step in the cross validation procedure.

### Due date for submission of this assignment is 11:59 pm Jun 14, 2018.
