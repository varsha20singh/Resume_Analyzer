import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK resources
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")

# Page Setup

st.set_page_config(page_title="Resume Scorer", page_icon="V",layout="wide")
st.write("""
        Upload your resume (PDF) and paste a job description to see how well
        they match!
        \nThis tool uses **TF-IDF + Cosin Similarity** to analyze your resume against job
        requirements.
            
""")
with st.sidebar:
    st.header("About")
    st.info("""
        This tool helps you :
        - Measures how well your resume matches your job description.
        - Identify important job keywords
        - Improve your resume based on missing terms.
""")
    st.header("How it works")
    st.write("""
        1. Upload your resume (PDF).
        2. Paste the job description.
        3. Click on **Analyze Match**.
        4. Review score and suggestion.   
""")
    
# Functions

def extract_text_from_pdf(upload_file):
    try:
        pdf_reader = PyPDF2.PdfReader(upload_file)
        txt = ""
        for page in pdf_reader.pages:
            txt = txt+page.extract_text()

        return txt
    except Exception as e:
        st.error(f"Error reading pdf : {e}")
        return ""
    
def clean_txt(txt):
    txt = txt.lower()
    txt = re.sub(r'[^a-zA-Z\s]','',txt)
    txt = re.sub(r'\s',' ',txt).strip()
    return txt

def remove_stopwords(txt):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(txt)
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_txt, job_description):
    resume_processed  = remove_stopwords(clean_txt(resume_txt))
    description_processed = remove_stopwords(clean_txt(job_description))
    vectorizer = TfidfVectorizer()
    tfidfmatrix  = vectorizer.fit_transform([resume_processed,description_processed])
    score = cosine_similarity(tfidfmatrix[0:1],tfidfmatrix[1:2])[0][0]*100
    return round(score,2)



def main():
    upload_file = st.file_uploader("Upload your resume (PDF)",type=['pdf'])
    job_description = st.text_area("Paste your job description", height=200)

    if(st.button("Analyze Score")):
        if not upload_file:
            st.warning("Please upload your resume")
            return
        if not job_description:
            st.warning("Please paste the job description.")
            return
        with st.spinner("Analyzing your resume..."):
            resume_txt = extract_text_from_pdf(upload_file)
            if not resume_txt:
                st.error("Could not extract text from the PDf, Please upload another one!")
                return
            
            similarity_score = calculate_similarity(resume_txt, job_description)
            st.subheader("Result")
            st.metric("Match score", f"{similarity_score}%")

            # Visualization 
            fig, ax = plt.subplots(figsize = (6,0.5))
            colors = ['Red','Yellow','Green']
            color_idx = min(int(similarity_score//33),2)
            ax.barh([0],[similarity_score],color = colors[color_idx])
            ax.set_xlim(0,100)
            ax.set_xticks([])
            ax.set_xlabel("Match Percentage")
            ax.set_title("Resume Job Match")
            st.pyplot(fig)

            if similarity_score<40:
                st.warning("Low Match !, improve your resume.")
            elif similarity_score>40 and similarity_score<70:
                st.info("Your resume align fairly well.")
            else:
                st.success("Excellent Match!, your resume is strongly aligned with the job description.")


if __name__=="__main__":
    main()




