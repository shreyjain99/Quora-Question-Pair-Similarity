
<h2 align= "center"><em>Quora Question Pair Similarity</em></h2>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/Quora-Question-Pair-Similarity/blob/main/src%20files/quora%20image.jpg"/>
</div>

<hr width="100%" size="2">

<h3 align= "left"> <b> Key Project Formulation </b> </h3>

<br>

<p>
<strong>Real World/Business Objective :</strong> To identify which questions asked on Quora are duplicates of questions that have already been asked doing this could be useful to instantly provide answers to questions that have already been answered.
</p>

<br>

<p>
<strong>Constraints :</strong>
</p>
<ol>
<li>High precision and recall </li>
<li>No strict latency constraints</li>
<li>Probabilistic output</li>
<li>Interpretability is partially important</li>
</ol>

<br>

<p>
<strong>Get the data from :</strong> https://www.kaggle.com/c/quora-question-pairs
<br>The data is hosted by Quora as a featured prediction competition on kaggle
</p>

<br>

<p>
<strong>Data Overview :</strong>
<br>
<p> 
- Data will be in a file Train.csv <br>
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate <br>
- Size of Train.csv - 60MB <br>
- Number of rows in Train.csv = 404,290
</p>

<br>


id - the id of a training set question pair
qid1, qid2 - unique ids of each question (only available in train.csv)
question1, question2 - the full text of each question
is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

<p>
<strong>Features in the dataset :</strong>
</p>
Dataset contains 404290 rows. The columns in the table are:<br />
<pre>
<b>Id</b> - the id of a training set question pair<br />
<b>qid1, qid2</b> - unique ids of each question (only available in train.csv)<br />
<b>question1, question2</b> - the full text of each question<br />
<b>is_duplicate</b> - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.<br />
</pre>

<br />

<br>

<p>
<strong>ML Problem Formulation :</strong>
</p>
<p> <strong>It is a multi-label classification problem</strong> </p>
<p> 
<b>Multi-label Classification</b>: Multilabel classification assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A question on Stackoverflow might be about any of C, Pointers, FileIO and/or memory-management at the same time or none of these. <br>
Credit: http://scikit-learn.org/stable/modules/multiclass.html
</p>

<br>
<br>

<p>
<strong>Performance metrics :</strong>
</p>
<ol>
<li>Micro-Averaged F1-Score (Mean F Score) </li>
<li>Hamming loss</li>
</ol>

<hr width="100%" size="2">

<br>

<body>

  <h3>Flow of Project : </h3>
  
  <br>

  <h3 align= "center"><strong>Data Loading</strong></h3>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center"><strong>Analysis of tags</strong></h3>
  <p align= "center"><em> - Total number of unique tags </em></p>
  <p align= "center"><em> - Number of times a tag appeared </em></p>
  <p align= "center"><em> - Tags Per Question </em></p>
  <p align= "center"><em> - Most Frequent Tags </em></p>
  <p align= "center"><em> - The top 20 tags </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Data Preprocessing (Natural Language Processing)</h3>
  <p align= "center"><em> - Separated out code-snippets from Body </em></p>
  <p align= "center"><em> - Removed Special characters from Question title and description </em></p>
  <p align= "center"><em> - Removed stop words (Except 'C') </em></p>
  <p align= "center"><em> - Converted all the characters into small letters </em></p>
  <p align= "center"><em> - Used SnowballStemmer to stem the words </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Machine Learning Models</h3>
  <p align= "center"><em> - Converted tags for multilabel problems </em></p>
  <p align= "center"><em>- Splited the data into test and train (80:20)  </em></p>
  <p align= "center"><em> - Featurized data (TFIDF FEATURES) </em></p>
  <p align= "center"><em> - Applyied Logistic Regression with OneVsRest Classifier</em></p>


  
</body>

<hr width="100%" size="2">
<br>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/TagGen-Automated-Tagging-for-Stack-Overflow-Questions/blob/main/src%20files/frequent%20tags.png"/>
</div>

<p>
<strong>Future Scope :</strong>
</p>
<ol>
<li>Use bag of words upto 4 grams and compute the micro f1 score with Logistic regression(OnevsRest) </li>
<li>Perform hyperparameter tuning on alpha (or lambda) for Logistic regression to improve the performance using GridSearch</li>
<li>Try OneVsRestClassifier with Linear-SVM (SGDClassifier with loss-hinge)</li>
</ol>

<hr width="100%" size="2">
