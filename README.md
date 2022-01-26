# Stacked-Model-for-Argument-Mining

In this project, we tackle the argument identification task on two different public datasets (Student Essays and Web Discourse), following two approaches: a classical machine learning SVM and DistilBert-based one. 
Moreover, this project sheds light on a new direction for researchers in this domain, since we validate the principle of ensemble learning. In other words, we show that combining multiple approaches via a well stacked model would improve the system performance. 

    Paper: https://link.springer.com/chapter/10.1007/978-3-030-86472-9_33
    
    
    
    To run the code from cmd: 
1) enter the location of your repository where the manager.py file exists.

2) run python manager.py train stack


To know how to enter other arguments and train svm or bert instead of the stacked model run:

python manager.py train svm

or

python manager.py train bert
