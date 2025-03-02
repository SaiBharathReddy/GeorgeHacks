import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import Tuple, List
import os
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)
import spacy
from openai import OpenAI

# Load environment variables
load_dotenv()

class CriticalityAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.intensifiers = ["heavily", "severely", "extremely", "critically", "intensely"] 
        self.emergency_patterns = {
            'cardiac_emergency': ['heart attack', 'chest pain', 'arrythmia'],
            'neuro_emergency': ['stroke', 'seizure', 'unconscious'],
            'respiratory_emergency': ['choking', 'asphyxia', 'respiratory distress'],
            'trauma_emergency': ['major hemorrhage', 'compound fracture', 'penetrating injury'],
            'mental_health_crisis': ['suicidal ideation', 'psychotic episode', 'severe dissociation']
        }
        self.symptom_ontology = {
            'cardiac': ['chest tightness', 'palpitation', 'syncope'],
            'respiratory': ['dyspnea', 'tachypnea', 'hypoxia'],
            'neurological': ['paresthesia', 'ataxia', 'aphasia'],
            'metabolic': ['polyuria', 'polydipsia', 'hyperglycemia']
        }

    def analyze_criticality(self, query: str) -> Tuple[float, str, List[str]]:
        doc = self.nlp(query.lower())
        emergency_score = self._calculate_emergency_score(doc)
        symptom_categories = self._detect_symptom_categories(doc)
        return self._determine_urgency(emergency_score, symptom_categories)

    def _calculate_emergency_score(self, doc) -> float:
        scores = []
        for category, patterns in self.emergency_patterns.items():
            pattern_docs = [self.nlp(pattern) for pattern in patterns]
            scores.extend([doc.similarity(pattern_doc) for pattern_doc in pattern_docs])
        
        base_score = max(scores) if scores else 0.0
        intensifier_boost = 0.1 * sum(1 for token in doc if token.text in self.intensifiers)
        
        return min(base_score + intensifier_boost, 1.0)

    def _detect_symptom_categories(self, doc) -> List[str]:
        categories = []
        for category, symptoms in self.symptom_ontology.items():
            symptom_docs = [self.nlp(symptom) for symptom in symptoms]
            if any(doc.similarity(symptom_doc) > 0.7 for symptom_doc in symptom_docs):
                categories.append(category)
        return categories

    def _determine_urgency(self, score: float, categories: List[str]) -> Tuple[float, str, List[str]]:
        if score > 0.75:
            return (score, "EMERGENCY: Immediate medical attention required", categories)
        elif score > 0.65:
            return (score, "URGENT: Seek care within 2 hours", categories)
        elif score > 0.45:
            return (score, "ACUTE: Schedule same-day evaluation", categories)
        else:
            return (score, "ROUTINE: Monitor and follow up if needed", categories)

def init_session_state():
    """Initialize all required session state variables"""
    required_keys = {
        'show_followup': False,
        'chat_history': [],
        'followup_input': '',
        'followup_submitted': False,
        'analyzer': CriticalityAnalyzer(),
        'provider': 'Deep-Seek'
    }
    
    for key, default in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def initialize_llm_client(provider: str):
    @st.cache_resource
    def _get_client(provider_name):
        api_key = "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"
        model_name = "deepseek/deepseek-r1:free"
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key), model_name
    return _get_client(provider)

def generate_response(query: str, client, model_name: str) -> str:
    system_prompt = """
**Medical Assistant Protocol**  

**Role**: Expert medical advisor providing accurate, concise guidance for health inquiries  

**Response Structure**:  
1. **Direct Answer** (2-3 sentences maximum)  
2. **First Response Steps** (**bold header**)  
   - Prioritized bullet points of immediate actions  
3. **Precautions** (**bold header**)  
   - Bullet points of warnings/avoidances  
4. **When to Seek Care** (**bold header**)  
   - Clear indicators for professional medical help  

**Format Rules**:  
- Use **bold** for section headers  
- Maintain bullet points (no numbering)  
- Keep responses under 150 words total  
- Use simple language (â‰¤8th grade level)  
- Include metric and imperial units where relevant  

**Non-Medical Query Response**:  
"I specialize in medical guidance. For other topics, please ask a different question."  

**Language Matching**:  
Maintain the user's original language for queries  
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def generate_followup_response(history: List[dict], new_query: str, client, model_name: str) -> str:
    context_prompt = "\n".join([f"Q: {item['query']}\nA: {item['response']}" for item in history])
    system_prompt = f"Act as  Expert medical advisor providing accurate, concise guidance for health inquiries and answer Previous conversation:\n{context_prompt}\n\nNew Response Requirements:\n1. Address both history and new query\n2. Update recommendations"
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": new_query}
            ],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating follow-up response: {str(e)}")
        return None

def display_enhanced_response(response: str, categories: List[str], show_resources: bool, show_tips: bool):
    st.write("🤔 Response:")
    st.write(response)
    if categories:
        st.write("⚕️ Related Categories:")
        for category in categories:
            st.write(f"- {category.capitalize()}")
    if show_resources:
        st.write("📚 Resources:\n- NHS\n- CDC\n- WHO")
    if show_tips:
        st.write("💡 Tips:\n- Consult professionals\n- Track symptoms")

def store_in_history(query: str, response: str, categories: List[str]):
    st.session_state.chat_history.append({
        'query': query,
        'response': response,
        'categories': categories,
        'timestamp': datetime.now()
    })

def collect_feedback(query, response, categories):
    col1, col2, col3 = st.columns(3)
    with col1: st.button("👍 Helpful")
    with col2: st.button("👎 Not Helpful")
    with col3: st.button("❌ Professional Help")

def clear_followup():
    st.session_state.followup_input = ""  # Resets input on every change

followup_query = st.text_input(
    "Enter follow-up question:", 
    key="followup_input",
    on_change=clear_followup  # Triggers on every keystroke
)

def handle_followup():
    if st.session_state.show_followup:
        with st.container():
            # Unique key using interaction count
            key_suffix = len(st.session_state.chat_history)
            followup_query = st.text_input(
                "Enter follow-up question:", 
                key=f"followup_input_{key_suffix}",  # Dynamic key
                help="Type additional questions here"
            )
            
        if st.button("Submit Follow-up", key=f"submit_btn_{key_suffix}"):  # Removed the disabled condition
            process_followup(followup_query)


def process_followup(query: str):
   with st.spinner('Processing follow-up question...'):
    """Process follow-up query and update history"""
    client, model_name = initialize_llm_client(st.session_state.provider)
    response = generate_followup_response(
        st.session_state.chat_history,
        query,
        client,
        model_name
    )
    
    if response:
        st.write("### Follow-up Response")  # Check if this gets printed
        st.write(response)  # Check if the response is not None or empty
        st.session_state.chat_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now()
        })
        st.session_state.conversation_context = st.session_state.chat_history[-3:]
        st.session_state.followup_submitted = True
    else:
        st.error("No response generated for the follow-up query.")

def text_generation_section():
    """Main query processing with integrated follow-up handling"""
    init_session_state()
    
    # Main query form
    with st.form(key="main_query_form"):
        st.write("Describe your medical concern:")
        
        # Settings in expander
        with st.expander("⚙️ Settings"):
            st.session_state.show_resources = st.checkbox("Show Resources", True)
            st.session_state.show_tips = st.checkbox("Show Tips", True)
            enable_emergency = st.checkbox("Emergency Alerts", True)
        
        # Main input and submission
        query = st.text_input(
            "", 
            placeholder="Enter medical query...", 
            key="main_input",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Analyze")
        
        if submitted and query:
            # Criticality analysis
          with st.spinner('Analyzing criticality and generating response...'):
            analyzer = st.session_state.analyzer
            score, urgency, categories = analyzer.analyze_criticality(query)
            
            # Display urgency alerts
            if enable_emergency:
                alert_container = st.container()
                with alert_container:
                    if score >= 0.9: st.error(urgency)
                    elif score >= 0.7: st.warning(urgency)
                    elif score >= 0.4: st.info(urgency)
                    else: st.success(urgency)
            
            # Generate response
            client, model_name = initialize_llm_client("Deep-Seek")
            if client:
                response = generate_response(query, client, model_name)
                if response:
                    # Update session state
                    store_in_history(query, response, categories)
                    st.session_state.conversation_context = [{
                        'query': query,
                        'response': response,
                        'timestamp': datetime.now()
                    }]
                    
                    # Display response
                    display_enhanced_response(
                        response, 
                        categories,
                        st.session_state.show_resources,
                        st.session_state.show_tips
                    )
                    
                    # Activate follow-up section
                    st.session_state.show_followup = True

    # Follow-up controls (outside main form)
    if st.session_state.get('show_followup', False):
        handle_followup()
        
    # Conversation management buttons
    control_col1, control_col2 = st.columns([1, 4])
    with control_col1:
        if st.button("🔄 New Chat", help="Start new conversation"):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.session_state.show_followup = False
            st.rerun()
    
    # Display conversation history
    if st.checkbox("Show Conversation History"):
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Interaction {len(st.session_state.chat_history)-idx}: {chat['query'][:30]}..."):
                st.write(f"**Query:** {chat['query']}")
                st.write(f"**Response:** {chat['response']}")
                st.write(f"**Timestamp:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

