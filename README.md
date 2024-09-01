# Airline reporting powered with AI
### Purpose
The purpose of this Python program is to automate the gathering, reporting, and analysis of data from online airline reviews, and deliver the results directly to your email.

This Python program automates the process of generating reports based on airline review data scraped from the web. It uses topic modeling to extract common themes for different Net Promoter Score (NPS) categories.

The program trains the model on demand, which may take a considerable amount of time to run, especially if the number of airlines and pages to scrape is high.

Net promoter score is determined as so:
- 8 <= promoter
- 5 >= detractor
- 6-7 = Neutral

![workflow](https://github.com/user-attachments/assets/f65d7db7-3b27-4f8a-8108-0d481c890c4c)

The above depicts the entire workflow of the project.

### Files overview
In each of the files you can find the following:
-  scrape.py: Gathers reviews and relevant information like airline and rating from airlinequality.com. The script will prompt for the number of airlines and pages to scrape.
-  report.py: Generates a PDF report with visualizations using Matplotlib and ReportLab.
-  topic_modelling.py: Handles data cleaning and preparation. It conducts Latent Semantic Analysis (LSA) and Latent Semantic Indexing (LSI), along with fine-tuning to identify the best model based on the Coherence Score for each NPS category. Number of topics can be adjust for tuning, there is no promot for this.
-  report_topics.py: Takes the output of the best model for each NPS score and generates a PDF with the relevant data.
-  send_email.py: Sends an email with the generated PDFs attached.
-  main.py: Streamlines the running process. You can edit email-related information here.

### Important Notes:
- Inputs: Be mindful of inputs requested by scrape.py, such as the number of airlines and pages to scrape, and the Gmail app password in send_email.py. You can change other variables as needed.
Best Model Output: The best model output for each NPS score is saved in the best_topic_models folder.
- Gmail App Password: Ensure you have two-factor authentication (2FA) activated for Gmail to generate an app password.
- Requirements.txt: Contains all the relevant python libraries used.

### Example Outputs:
You can find example reports in the reports and report_topics folders, as well as the output of the scraping script in the data folder.
