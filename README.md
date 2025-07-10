I am using single knn model as my base for testing knn accuracy varies from 75% to 85%, i used dataset.py to divide the dataset and convert text based ans into numerical ( i didn't normalize it in data code because it was messing accuracy in furthe ensembles)
dataset.py also randomizes the dataset so that i can test the conditions effectively

I created 3 different ensembles 
random forest - accuracy - 90.1%~90.3% 
gradient boosting - accuracy - 90.1%~90.4
stacking without keras - accuracy - 90.9%~91.3%
(i also tried using neural network keras in stacking but performed worse than all of the above)
finally then i decided to try and make ensemble of all 3 models to try and get greater accuracy plus precision and recall

 i made a custom soft voting ensemble and tweaking with values i achieved best score of 92.7% accuracy
this model is in final_ensemble.py

Install required modules for running all the codes

<pre> pip install -r requirements.txt </pre> 

