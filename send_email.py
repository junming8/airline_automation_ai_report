import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os

def send_latest_reports_email(sender_email, receiver_email, password, subject, body):
    def get_latest_report(directory):
        if not os.path.isdir(directory):
            print(f"{directory} does not exist.")
            return None
        
        files = os.listdir(directory)
        pdf_files = [f for f in files if f.endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {directory}.")
            return None
        
        # Get the most recently modified file
        latest_file = max(pdf_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        return os.path.join(directory, latest_file)
    
    def send_email_with_attachments(sender_email, receiver_email, password, subject, body, attachments):
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"

        # Create a MIMEMultipart object
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = subject

        # Attach the email body
        message.attach(MIMEText(body, 'plain'))

        # Attach each file
        for attachment_path in attachments:
            if attachment_path:
                with open(attachment_path, "rb") as attachment:
                    part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                    message.attach(part)

        # Create a secure SSL context
        context = ssl.create_default_context()

        # Send email
        try:
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, message.as_string())
                print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")

    # Get the latest report from both directories
    reports_dir = 'reports'
    topics_dir = 'report_topics'
    report_paths = [
        get_latest_report(reports_dir),
        get_latest_report(topics_dir)
    ]

    # Send the email with both attachments
    if any(report_paths):
        send_email_with_attachments(sender_email, receiver_email, password, subject, body, report_paths)
    else:
        print("No reports found to send.")


