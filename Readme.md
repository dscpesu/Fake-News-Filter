I have uploaded four files
The dataset containing the titles of 69 articles. The columns are separated by tabs and not commas
 
A python file named fakenews.py which uses naive-bayes classification method to determine 
if the title is fake or not and also contains the basic pre-processing of the text.This 
does not give a good accuracy(58%).

A python file named tfidf implementation.

And another python file named tfidf-fakenews which implements the tfidf model on the dataset 
of article headings. It shows the tfidf scores of all the words in the dataset.
However I haven't been able to use these scores in predicting if it is fake or not.It still
uses naive bayes classification model. So the tfidf table exists we need to use those values 
in predicting the final result now.

