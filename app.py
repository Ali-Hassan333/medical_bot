from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import pytesseract
import pdfplumber
from openai import OpenAI


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
client = OpenAI(api_key="")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    structured_data = []
    
    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            text = extract_text_from_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        structured_data.append(process_text_with_llm(text))
    
    return render_template('index.html', structured_data=structured_data)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()
    return text

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def process_text_with_llm(text):
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "user", "content": f"""I am sharing a medical lab report via a {text}, and I need you to produce a comprehensive health analysis based on the uploaded lab report. The analysis should take the form of an easily understood, coherent, well-laid-out report that should include, but not be limited to, various metrics such as cholesterol levels, glucose levels, kidney function, electrolyte balance, liver function, prostate health, inflammatory markers, and vitamin/mineral levels. Please provide each detailed health report based only on the data from the specific lab results provided to you.

Please note that I fully understand that different user reports may focus on different types of tests whether from blood, urine, stool, or saliva, etc. I recognize that lab reports could be drawn from any one of these test types, or any combination of test types, or all types on a single lab report (e.g., full medical lab report). It is therefore expected that your report should include as many metrics as permitted based on the type of test and the source data provided. Where particular data may not have been captured on the report under consideration, those metrics would not be expected to appear in your output health analysis report. Please do your best to produce as rich an analysis as possible, whatever the type of lab report that is provided to you.

Key Metrics for the output report could include the following where applicable:

Summary of Key Health Metrics:

List each metric, its value, the optimal range, the standard range, and an interpretation of whether it is normal, high, or low.
Detailed Analysis & Recommendations:

For each metric, provide a detailed explanation of what the value means for my health.
Offer personalized dietary, lifestyle, and supplement recommendations. Please infuse recommended dietary changes and specific herbs that can be used to address specific health concerns based on widely available data that you can reference. Place a special focus on highly acclaimed Jamaican herbs. However, do not limit your herbal recommendations to Jamaican herbs only. If there are globally accepted herbal remedies that can be equally effective, you may recommend them also.
Incorporate an Alkaline Whole Foods approach to healing the body in these recommendations.
Suggest any further tests or follow-up actions that might be needed for each identified concern.
Health Risk Assessment:

Evaluate the risk levels for all possible chronic conditions such as diabetes, cardiovascular diseases, liver diseases, kidney diseases, and cancer based on the lab results.
Tabular Insights:

Include numerical spreadsheet charts, where practical for key metrics, particularly for cholesterol levels, glucose levels, kidney function, electrolytes balance, liver function, prostate health, inflammation levels, iron levels, and vitamin D levels, etc.
Trends and Observations:

Identify any trends in the data and their implications for the user's health.
Probability of Dysfunction:

Assess the probability of dysfunction in areas such as lipid panel, inflammation, acid-base balance, toxicity, heavy metals, and oxidative stress, etc.
What to Ask Your Doctor on Your Next Visit:

Provide a comprehensive list of specific, pointed questions I should ask my doctor based on the analysis of the lab results.
Break down this section into subheads that identify each stated health concern, and produce a set of specific questions for the doctor, for each stated concern.
Personalized Health Insights:

Tailor a set of useful insights to the user's specific age, gender, and health status
Lifestyle and Environmental Factors:

Consider how lifestyle factors such as smoking, alcohol consumption, physical activity, and environmental exposures may impact the lab results and overall health, and make relevant recommendations towards more positive outcomes
Genetic Predispositions:

If genetic data is available or mentioned, include insights on how genetic predispositions might affect the lab results and health risks.
Mental Health Indicators:

Assess potential indicators of mental health conditions such as stress, anxiety, and depression from relevant biomarkers if present in the lab report.
Immune System Function:

Evaluate the status of the immune system through markers such as white blood cell count and specific immune-related proteins.
Hydration Status:

Analyze hydration levels using metrics like blood urea nitrogen (BUN) and electrolytes.
Bone Health:

Include metrics related to bone health such as calcium levels, vitamin D levels, and other relevant markers.
Hormonal Balance:

Evaluate hormonal levels, including thyroid function, sex hormones, and adrenal function if relevant data is provided.
Gut Health:

Assess markers related to gut health and digestive function if stool tests or related data are available.
Possible Causes. For each health issue that is identified, you should add a paragraph that comprehensively outlines the possible causes of the issue, including eating habits, lifestyle, genetics etc.
Implications of Procrastination  For each health issue that is identified, you should add a paragraph that comprehensively outlines  the real implications that exist, should the patient fail to act on the results and recommendations provided."""},
        ],
        
        # max_tokens=1000,
        
        
    )
    return response.choices[0].message.content,

if __name__ == '__main__':
    app.run(debug=False)