import streamlit as st
from text_generation import text_generation_section
from dotenv import load_dotenv
from second_opinion import second_opinion_section
from image_analysis import image_analysis_section

# Load environment variables
load_dotenv()

def main():
    st.title("🏥 Enhanced Medical AI Bot")
    st.write("Toggle between text-based generation and image-based analysis.")

    # Main switch to select functionality
    option = st.radio("Choose functionality", ["Talk to AI", "Second Opinion","Image Analysis"])

    # Display relevant section based on the chosen functionality
    if option == "Talk to AI":
        st.write("Text Generation Section")
        text_generation_section()  
    elif option == "Second Opinion":
        st.write("Second Analysis Section")
        second_opinion_section()
    elif option == "Image Analysis":
        st.write("Image Analysis Section")
        image_analysis_section()  

if __name__ == "__main__":
    main()
