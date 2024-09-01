# airline_report
Python program that makes use of automation to generate reports based on airlines review data that were scraped from the web. Topic modelling done on data to extract common topics for different NPS(Net promoter score) scores.

Net promoter score is determined as so:
- 8 >= promoter
- 5 >= detractor
- 6-7 = Neutral

![workflow](https://github.com/user-attachments/assets/f65d7db7-3b27-4f8a-8108-0d481c890c4c)

The above depicts the entire workflow of the project.

In each of the files you can find the following:
-  scrape.py: Used to gather reviews and their releavnt information like airline and rating from https://www.airlinequality.com/. Will ask for how many airlines and pages to scrape for.
-  report.py: Generate a pdf report including visualisation using matplotlib and reportlab to achieve so
-  topic_modelling.py: Includes the cleaning and preparations for data. Then conduct LSA and LSI along with fine tuning to get the best model based on the Coherance Score for each of the NPS score. You can change the tuning accordingly (Number of topics).
-  report_topics.py: Take teh output of the best model for each NPS score and write to a pdf with the relevant data
-  send_email.py: Send an email with both the pdfs attached
-  main.py: A python file to streamline the running process, can edit the information regarding the email here.

Should look out for the inputs that asks for how much to scrape for in scrape.py and the input of app password for gmail. Rest of the variables one can change them accordingly.

*Note for app password for gmail you have to have 2FA activated.
