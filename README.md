# Email Topic Modeling Project

## Introduction
This project attempts to solve the email inbox overload problem. Inboxes should be managed properly in order to idenfity priorities and close tasks properly. Too often, important time-sensitive email can get lost in spam, junk, and irrelevant messages. What if we could utilize unsupervised machine learning to help us manage our inbox, outside of your typical spam handler, rules, and followup management.

We will utilize topic modeling, more specifically Latent Dirichlet Allocation (LDA), to help organize message to specific bins. The LDA model assigns probabilities to the bins that the messages most closely resemble amongst all the other messages in the corpus. Since this is unsupervised learning approach, we need a human perspective to help classify the meanings of the similar bins of messages. 

There are 2 main components to the model including the machine learning pipeline, and the demo frontend application. The ML pipeline scripts are in Jupyter format, with the primary email data extracted from the Gmail API and organized in a pandas dataframe. The front-end AI resides in a Docker container for transferable deployment.

# Content
1) Model
  - Extract, Transform, Load (ETL)
  - Preprocess
  - Exploratory Data Analysis (EDA) / Training
  - Inference
2) App
  - Main app

# Technologies
- Python
- Docker
- VSCode
- Streamlit
- GitHub

# Launch Date
- 10/29/2022

# Project status 
The project is ready for review.

# Results
Based on my email extraction data, 3 meaninguful topics were derived from the message content, and identified with the LDA model. Since this is an unsupervised approach, it is difficult to measure the metric results. We perform inference on the test dataset and see if the overall direction of the assigned topics align with the content of the messages. The messages seem to fit pretty well with a few looking to be misappropriated to ambiguous buckets.
We can then visualize the expected outputs with the demo UI, contained from the app.py script. The PowerPoint presentation will present more findings and details related to the project.
