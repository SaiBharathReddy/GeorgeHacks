import streamlit as st
from PIL import Image
from text_generation import initialize_llm_client
from text_generation import generate_response
# from text_generation import store_in_history
from text_generation import collect_feedback


def second_opinion_section():
    """Section for getting second opinion on diagnosis"""
    with st.form("second_opinion_form"):
        st.write("### 📝 Patient Information")
        user_problem = st.text_area("Describe your symptoms/medical problem:")
        doctor_diagnosis = st.text_input("Enter doctor's diagnosis/prescription:")
        
        submitted = st.form_submit_button("Analyze Diagnosis")
        
    if submitted and user_problem and doctor_diagnosis:
        with st.spinner("🔍 Analyzing diagnosis..."):
            analysis_prompt = f"""Analyze this medical case:
Patient Problem: {user_problem}
Doctor's Diagnosis: {doctor_diagnosis}

Evaluate whether the diagnosis is appropriate considering:
1. Symptom-match with known conditions
2. Typical diagnostic criteria
3. Possible alternative diagnoses
4. Medication appropriateness (if prescribed)

Format response as:
- Correctness Assessment: [Match Level]
- Reasoning: [Detailed Analysis]
- Recommendations: [Sugessions]"""

            client, model_name = initialize_llm_client("Groq")
            response = generate_response(analysis_prompt, client, model_name)
            
            if response:
                st.write("### 📋 Analysis Results")
                st.markdown(response)
                
                # Store interaction
                # store_in_history(
                #     f"Problem: {user_problem}\nDiagnosis: {doctor_diagnosis}", 
                #     response,
                #     ["Second Opinion"]
                # )
                collect_feedback(
                    f"Problem: {user_problem}\nDiagnosis: {doctor_diagnosis}",
                    response,
                    ["Second Opinion"]
                )
            else:
                st.error("Failed to analyze diagnosis. Please try again.")