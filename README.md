# Rating_Prediction
How do linear and non-linear predictive models perform in predicting the quality ratings of AI-generated presentations, and which features contribute most significantly to these predictions?

In this repository you can find all of the necessary files (excel as well as python) to recreate the research that I conducted during my Internship stay at Sendsteps.ai. 
Please refer to the attached PDF file entailing a full report of the research steps taken to fulfill the goal of:

  **A.** Identify which features are most strongly correlated to the presentation rating
  
  **B.** Leverage machine learning techniques to predict presentation ratings

You can find two excel files that contain data.

The first file **dataset3099.csv** represents the raw data used for this project [3099 presentations] collected by a SQL query (referring to file SQLquery in repository). 
List of the features:


  **rating**, which represents the target variable [1 to 5];
	
  **numSlides**, number of presentation slides [Integer];
 
  **numShapes**, number of presentation elements [Integer];
	
  **isEdited**, if generated presentation was edited or not [Binary];
	
  **hasVideo**, if presentation has a video or not [Binary];
	
  **hasWordcloud**, if presentation has a word-cloud [Binary];
	
  **hasMPC**, if presentation has quiz [Binary];
	
  **avgX**, average X coordinate of all shapes in presentation [Continuous];
	
  **avgY**, average Y coordinate of all shapes in presentation [Continuous];
	
  **avgWidth**, average width value of all shapes in presentation [Continuous];
	
  **avgHeight**, average height value of all shapes in presentation [Continuous];
  
  **presentationStyleId**, presentation template [Integer];
  
  **initialPromptTokens**, character length of initial prompt send to LLM [Integer];
  
  **responseTokens**, character length of LLM response [Integer];
  
  **length**, structure of presentation content [Text];
  
  **toneOfVoice**, tone of presentation content [Text];
  
  **backgroundOpacity**, opacity of presentation background [Continuous];
  
  **imgURLs**, link of images of presentation [Text];
  
  **subject**, presentation subject [Text];
  
  **language**, presentation language [Text]

  

The second datafile **afterSCALING.xlsx** [7099 presentations] is representative of the same data as the previous file (**dataset3099.xlsx**) after preprocessing the data (e.g. feature selection, oversampling using SMOTENC, one-hot encoding, feature sampling, among others - view the full report for more details)


Additionally, this reportistory entails 6 python files.

**SMOTENC.py** is the code that was used to oversample the imbalanced raw data using the SMOTENC oversampling technique.

**CLIP.py** is the code to create the new feature 'imgRelevance' using OpenAIs CLIP Model. This score represents the average relevance of presntation images to the presentation subject.

**GB.py** and **RFC.py** represent the code for the linear models: Gradient Boosting Classifier and Random Forest Classifier + built-in feature ranking (Gini importance).

**bayesian.py** is the code for the Neural Network, using bayesian hyperparameter optimization (hence the name of the file).

Lastly, **scaling.py** is the python file that was used to apply one-hot encoding as well as standardization/normalization to the data.


Evidently there have been more intermediate steps, as can be observed when reading through the report (), but for the sake of cluttering and bas oversight, these messy python files are left out of this repository. These steps can easily be reproduced (e.g. plots, elimination or combination of features, and so on).
