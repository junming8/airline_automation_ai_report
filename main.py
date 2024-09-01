from scrape import scrape_airline_reviews
from report import generate_pdf_report
from send_email import send_latest_reports_email
from topic_modelling import topic_modelling_function
from report_topics import write_results_to_pdf
import os

# scraped_data = scrape_airline_reviews()

# generate_pdf_report(scraped_data)
if __name__ == '__main__':
    scraped_data = scrape_airline_reviews()

    generate_pdf_report(scraped_data)

    topic_modelling_function()

    json_file = 'best_topic_models.json'
    output_dir = 'report_topics'
    os.makedirs(output_dir, exist_ok=True)
    pdf_file = os.path.join(output_dir, 'topic_modeling_results.pdf')
    write_results_to_pdf(json_file, pdf_file)


    # Replace these with actual values
    sender_email = "enter sender email"
    receiver_email = "enter reciever email"
    password = input("Type your app password and press enter: ")
    subject = "Latest Airline Reviews Report"
    body = '''Hi,

    I hope this email finds you well. Please refer to the latest reports attached to this email:

        - Latest Report**: This report provides the most recent data and insights.
        - Topic Modeling Results**: Includes key words used for each NPS score category.

    For an overview report of the general consensus of the airline market, such as ratings, refer to the airline_reviews_report.

    Please refer to the topic_modelling_results PDF for key words used for each NPS score category.

    Best Regards,
    Automation Script'''

    send_latest_reports_email(sender_email, receiver_email, password, subject, body)