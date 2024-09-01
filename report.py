import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime
import os

def generate_pdf_report(data):
    airlines = data['airline_name'].unique()
    
    # Create the reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Generate filename with date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join('reports', f'airline_reviews_report_{timestamp}.pdf')
    
    # Create a PDF canvas
    c = canvas.Canvas(pdf_path, pagesize=letter)
    page_width, page_height = letter  # Page dimensions in points
    
    def create_summary_charts():
        # Pie Chart - Overall Recommendation Distribution
        recommendation_data = data['recommended'].value_counts()
        plt.figure(figsize=(4, 4))
        plt.pie(recommendation_data, labels=None, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
        plt.title('Overall Recommendation', pad=20)
        plt.legend(labels=recommendation_data.index, loc="center left", bbox_to_anchor=(1, 0.5))  # Legend on the right
        plt.axis('off')
        pie_chart_path = 'overall_pie_chart.png'
        plt.savefig(pie_chart_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Pie Chart - Overall NPS Score Distribution
        nps_data = data['NPS_score'].value_counts()
        plt.figure(figsize=(4, 4))
        plt.pie(nps_data, labels=None, autopct='%1.1f%%', colors=['#ff9999','#66b3ff', '#ffcc99'])
        plt.title('Overall NPS Score', pad=20)
        plt.legend(labels=nps_data.index, loc="center left", bbox_to_anchor=(1, 0.5))  # Legend on the right
        plt.axis('off')
        nps_chart_path = 'overall_nps_pie_chart.png'
        plt.savefig(nps_chart_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Line Graph - Number of Reviews Over Time by NPS Score
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        monthly_reviews = data.groupby([data['date'].dt.to_period('M'), 'NPS_score']).size().unstack(fill_value=0)
        plt.figure(figsize=(10, 6))
        monthly_reviews.plot(kind='line', marker='o')
        plt.title('Overall Monthly Reviews Trend')
        plt.xlabel('Month')
        plt.ylabel('Number of Reviews')
        plt.legend(title='NPS Score')
        plt.tight_layout()  # Adjust layout to fit labels
        line_chart_path = 'overall_line_chart.png'
        plt.savefig(line_chart_path, bbox_inches='tight')  # Save without extra whitespace
        plt.close()
        
        return pie_chart_path, nps_chart_path, line_chart_path

    def draw_title(airline_name=None):
        c.setFont("Helvetica", 12)
        title_text = f'Airline: {airline_name} Review Report' if airline_name else 'Overall Airlines Review Report'
        c.drawString(100, page_height - 20, title_text)
        c.drawString(100, page_height - 40, f'Report Date: {datetime.datetime.now().strftime("%Y-%m-%d")}')
    
    # Create summary charts
    overall_pie_chart, overall_nps_chart, overall_line_chart = create_summary_charts()
    
    # Draw the first page with the overall summary
    draw_title()
    
    # Calculate positions dynamically
    title_height = 60
    chart_width = 200
    chart_height = 200
    gap = 20
    
    # Insert Overall Recommendation Pie Chart
    pie_image = ImageReader(overall_pie_chart)
    pie_x = 75
    pie_y = page_height - title_height - chart_height - gap
    c.drawImage(pie_image, pie_x, pie_y, width=chart_width+5, height=chart_height)
    
    # Insert Overall NPS Score Pie Chart
    nps_image = ImageReader(overall_nps_chart)
    nps_x = pie_x + chart_width + gap
    c.drawImage(nps_image, nps_x, pie_y, width=chart_width+35, height=chart_height)
    
    # Insert Overall Line Chart
    line_image = ImageReader(overall_line_chart)
    line_width = page_width - 150
    line_height = 360
    line_x = 75
    line_y = pie_y - line_height - gap
    c.drawImage(line_image, line_x, line_y, width=line_width, height=line_height)
    
    c.showPage()

    # Remove summary PNG files after adding them to the PDF
    os.remove(overall_pie_chart)
    os.remove(overall_nps_chart)
    os.remove(overall_line_chart)
    
    # Start a new page for the airline reports

    # Individual Airline Reports
    for airline in airlines:
        draw_title(airline)
        # Pie Chart - Distribution of Recommended vs Not Recommended
        recommendation_data = data[data['airline_name'] == airline]['recommended'].value_counts()
        plt.figure(figsize=(4, 4))
        plt.pie(recommendation_data, labels=None, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
        plt.title('Recommendation', pad=20)
        plt.legend(labels=recommendation_data.index, loc="center left", bbox_to_anchor=(1, 0.5))  # Legend on the right
        plt.axis('off')
        pie_chart_path = f'{airline}_recommendation_pie_chart.png'
        plt.savefig(pie_chart_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # NPS Score Pie Chart
        nps_data = data[data['airline_name'] == airline]['NPS_score'].value_counts()
        plt.figure(figsize=(4, 4))
        plt.pie(nps_data, labels=None, autopct='%1.1f%%', colors=['#ff9999','#66b3ff', '#ffcc99'])
        plt.title('NPS Score', pad=20)
        plt.legend(labels=nps_data.index, loc="center left", bbox_to_anchor=(1, 0.5))  # Legend on the right
        plt.axis('off')
        nps_chart_path = f'{airline}_nps_pie_chart.png'
        plt.savefig(nps_chart_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()


        # Insert Pie Charts Side by Side
        pie_image = ImageReader(pie_chart_path)
        nps_image = ImageReader(nps_chart_path)
        pie_width = 200
        pie_height = 200
        gap = 20
        pie_x = 75
        pie_y = page_height - 60 - pie_height - gap
        c.drawImage(pie_image, pie_x, pie_y, width=pie_width+5, height=pie_height)
        c.drawImage(nps_image, pie_x + pie_width + gap, pie_y, width=pie_width+35, height=pie_height)
        
        # Line Graph - Number of Reviews Over Time by NPS Score
        airline_data = data[data['airline_name'] == airline]
        airline_data['date'] = pd.to_datetime(airline_data['date'], errors='coerce')
        monthly_reviews = airline_data.groupby([airline_data['date'].dt.to_period('M'), 'NPS_score']).size().unstack(fill_value=0)
        plt.figure(figsize=(10, 6))
        monthly_reviews.plot(kind='line', marker='o')
        plt.title(f'Monthly Reviews Trend for {airline}')
        plt.xlabel('Month')
        plt.ylabel('Number of Reviews')
        plt.legend(title='NPS Score')
        plt.tight_layout()  # Adjust layout to fit labels
        line_chart_path = f'{airline}_line_chart.png'
        plt.savefig(line_chart_path, bbox_inches='tight')  # Save without extra whitespace
        plt.close()
        
        # Insert Line Chart
        line_image = ImageReader(line_chart_path)
        line_width = page_width - 150
        line_height = 360
        line_x = 75
        line_y = pie_y - line_height - gap
        c.drawImage(line_image, line_x, line_y, width=line_width, height=line_height)

        c.showPage()

        # Remove PNG files after adding them to the PDF
        os.remove(pie_chart_path)
        os.remove(nps_chart_path)
        os.remove(line_chart_path)
        
    # Save PDF
    c.save()
    
    print(f"PDF report saved as: {pdf_path}")
