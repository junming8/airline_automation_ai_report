import json
from fpdf import FPDF
import os

# Define a function to generate the PDF
def write_results_to_pdf(json_file, pdf_file):
    # Create instance of FPDF class
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Read the JSON file
    with open(json_file, 'r') as file:
        results = json.load(file)
    
    # Add a title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Topic Modeling Results", ln=True, align='C')
    pdf.ln(10)  # Add a line break
    
    # Add results to PDF
    pdf.set_font("Arial", size=12)
    for label, result in results.items():
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"{label}", ln=True, align='L')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Best Model Type: {result['best_model_type']}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Number of Topics: {result['best_num_topics']}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Coherence Score: {result['best_coherence_score']:.4f}", ln=True, align='L')
        
        pdf.ln(5)  # Add a small line break
        
        # Add topics
        pdf.set_font("Arial", 'I', 12)
        for topic in result['topics']:
            pdf.multi_cell(0, 10, txt=topic[1])  # Use multi_cell to handle long text
            pdf.ln(2)
        
        pdf.ln(10)  # Add a larger line break between segments
    
    # Save the PDF
    pdf.output(pdf_file)

