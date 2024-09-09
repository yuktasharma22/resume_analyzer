import streamlit as st
import pickle
import nltk
from PyPDF2 import PdfReader  # Correct import for PdfReader
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# PDF extraction function
def extract_text_from_pdf(upload_file):
    pdf_reader = PdfReader(upload_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to clean resume
def cleanResume(resume_text):
    cleanTxt = re.sub('http\S+\s', ' ', resume_text)
    cleanTxt = re.sub('RT|cc', ' ', cleanTxt)
    cleanTxt = re.sub('@\S+', ' ', cleanTxt)
    cleanTxt = re.sub('#\S+', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$&%'()*+,-./:;<=>'?@[\]^_{|}"""), ' ', cleanTxt)
    cleanTxt = re.sub(r"(?<=\w)'(?=\w)", '', cleanTxt)
    cleanTxt = re.sub(r'[^\x100-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    return cleanTxt

# Web app
def main():
    st.title("Resume Screening App")
    
    upload_file = st.file_uploader('Upload resume', type=['txt', 'pdf'])

    if upload_file is not None:
        # Check if the file is a PDF
        if upload_file.name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(upload_file)  # Extract text from PDF
        else:
            resume_bytes = upload_file.read()
            try:
                resume_text = resume_bytes.decode('utf-8')  # Handle text file
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

        # Clean the extracted resume
        cleaned_resume = cleanResume(resume_text)
        
        # Show cleaned resume for debugging (optional)
        st.write("Cleaned Resume Text:", cleaned_resume)

        # Transform the cleaned resume into features
        cleaned_resume = tfidf.transform([cleaned_resume])
        
        # Make prediction using the classifier
        prediction_id = clf.predict(cleaned_resume)[0]
        
        # Mapping predicted category ID to its corresponding name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        
        # Display the predicted category
        st.write(f"Predicted Category: {category_name}")

if __name__ == "__main__":
    main()

