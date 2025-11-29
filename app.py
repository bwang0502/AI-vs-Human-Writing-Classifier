"""
Streamlit App for AI vs Human Text Detection
Multi-Page App with Binary and Multi-Class Models
"""

import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import numpy as np
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set page config - this MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for font styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    
    /* Apply to all text - more aggressive targeting */
    * {
        font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    html, body, [class*="css"], div, p, span, label {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton>button, button {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Text input and text area */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, input, textarea {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] * {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown * {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Captions and labels */
    .stCaption, small, label {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Progress text */
    .stProgress > div > div > div {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        font-family: 'Montserrat', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Set style for plots
sns.set_style("whitegrid")

# Device setup - use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define example texts at module level (before main())
AI_EXAMPLES = [
    """The implementation of advanced methodologies facilitates the optimization of operational parameters, thereby enhancing overall system efficacy and performance metrics.""",
    
    """In order to achieve optimal outcomes, it is imperative to leverage synergistic approaches that maximize stakeholder engagement while simultaneously ensuring regulatory compliance across all operational domains.""",
    
    """The utilization of state-of-the-art technologies enables organizations to streamline their workflows, resulting in improved productivity and enhanced deliverable quality throughout the entire project lifecycle.""",
    
    """Contemporary research methodologies facilitate the examination of complex phenomena through systematic investigation, thereby contributing to the advancement of knowledge in various academic disciplines.""",
    
    """The integration of artificial intelligence systems represents a transformative paradigm shift that fundamentally alters traditional business processes and operational frameworks across multiple industry sectors."""
]

HUMAN_EXAMPLES = [
    """I went to the store yesterday and forgot my wallet. It was so embarrassing! Had to go back home and get it.""",
    
    """Can't believe it's already Friday. This week flew by so fast! Anyone else feel like they barely got anything done?""",
    
    """My dog keeps stealing socks from the laundry basket. Found like 10 of them under his bed today. He's such a goofball lol.""",
    
    """Just finished watching that new show everyone's talking about. Not gonna lie, the ending was pretty disappointing. Expected way more.""",
    
    """Tried making pasta from scratch for the first time. Total disaster. Somehow ended up with what can only be described as dough soup. Ordering pizza instead."""
]


@st.cache_resource  # This decorator caches the model so we don't reload it every time
def load_model():
    """
    Load the fine-tuned DistilBERT model.
    @st.cache_resource means this only runs ONCE, even if users interact with the app.
    """
    model_path = 'results/finetuned_improved_optionA/20251104_220909/distilbert_finetuned_improved.pt'
    
    # Load the checkpoint (saved model weights + metadata)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load the base DistilBERT architecture
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2  # Binary: Human (0) or AI (1)
    )
    
    # Load your trained weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode (turns off dropout, etc.)
    
    return model, checkpoint


@st.cache_resource
def load_tokenizer():
    """
    Load the tokenizer that converts text into numbers the model understands.
    Same tokenizer used during training.
    """
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def predict_text(text, model, tokenizer):
    """
    Make a prediction on the input text.
    
    Returns:
        prediction: 0 (Human) or 1 (AI)
        probabilities: [prob_human, prob_ai]
    """
    # Tokenize: Convert text to numbers
    encoding = tokenizer(
        text,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        max_length=128,           # Truncate if longer
        padding='max_length',     # Pad if shorter
        truncation=True,
        return_tensors='pt'       # Return PyTorch tensors
    )
    
    # Move to same device as model (GPU or CPU)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Run through model (no gradient calculation needed for inference)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Raw output scores
        
        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Get the predicted class (0 or 1)
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction, probabilities


def create_lime_explanation(text, model, tokenizer, num_features=10):
    """
    Generate LIME explanation showing which words influenced the prediction.
    
    LIME works by:
    1. Creating variations of the text (removing words)
    2. Seeing how predictions change
    3. Identifying which words matter most
    """
    
    # Create prediction function for LIME
    def predict_proba(texts):
        """LIME needs a function that takes text and returns probabilities."""
        probs = []
        for txt in texts:
            encoding = tokenizer(
                txt,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                probs.append(prob)
        
        return np.array(probs)
    
    # Create LIME explainer
    explainer = LimeTextExplainer(class_names=['Human', 'AI'])
    
    # Generate explanation
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,  # Top N most important words
        num_samples=500  # Number of variations to test
    )
    
    return exp


def plot_lime_explanation(exp, prediction):
    """
    Create a nice visualization of LIME results.
    Shows which words push toward AI vs Human.
    """
    # Get the explanation for the predicted class
    # Use the prediction directly as LIME label
    try:
        exp_list = exp.as_list(label=prediction)
    except KeyError:
        # If prediction label doesn't exist, try label 1 (the positive class)
        exp_list = exp.as_list(label=1)
    
    # Separate into words and weights
    words = [item[0] for item in exp_list]
    weights = [item[1] for item in exp_list]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars: green for positive (supports prediction), red for negative
    colors = ['green' if w > 0 else 'red' for w in weights]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(words))
    ax.barh(y_pos, weights, color=colors, alpha=0.6, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.set_xlabel('Impact on Prediction', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(words)} Most Influential Words', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Invert y-axis so most important is at top
    ax.invert_yaxis()
    
    return fig


# ============================================================================
# MAIN APP UI
# ============================================================================

def main():
    # Add page selection in sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Model",
            ["Binary Classifier", "Multi-Class Classifier"],
            help="Choose between detecting Human vs AI (binary) or identifying specific AI models (7-class)"
        )
        
        st.markdown("---")
        
        # Model Info
        st.header("Model Info")
        st.write(f"**Model:** DistilBERT (fine-tuned)")
        
        st.markdown("---")
        
        st.header("About")
        st.write("""
            This Streamlit app uses fine-tuned DistilBERT transformer models to detect AI-generated text.
            Choose between binary classification (Human vs AI) or multi-class identification (which specific AI model).
        """)
    
    # Route to appropriate page
    if page == "Binary Classifier":
        binary_classifier_page()
    else:
        multiclass_classifier_page()

def binary_classifier_page():
    """Original binary classification UI"""
    
    # Title and description
    st.title("AI vs Human Text Detector")
    st.markdown("""    
    **How it works:**
    1. Enter or paste any text
    2. The model analyzes linguistic patterns
    3. Get a prediction with confidence score
    4. See which words influenced the decision (LIME explanation)
    """)
    
    st.markdown("---")
    
    # Load model and tokenizer (cached, so only loads once)
    with st.spinner("Loading model..."):
        model, checkpoint = load_model()
        tokenizer = load_tokenizer()
    
    # Show model info and examples in sidebar
    with st.sidebar:
        st.markdown("---")
        
        st.header("Model Performance")
        st.write(f"**Training F1-Score:** {checkpoint['metrics']['f1']:.2%}")
        st.write(f"**Training Accuracy:** {checkpoint['metrics']['accuracy']:.2%}")
        
        st.markdown("---")
        
        st.header("Example Texts")
        # Add unique keys to prevent duplicate widget IDs
        if st.button("Load AI Example", key="load_ai_binary"):
            st.session_state.example_text = random.choice(AI_EXAMPLES)
        
        if st.button("Load Human Example", key="load_human_binary"):
            st.session_state.example_text = random.choice(HUMAN_EXAMPLES)
    
    # Main input area
    st.header("Enter Text")
    
    # Text input box (large, multi-line)
    text_input = st.text_area(
        label="Input Text",
        value=st.session_state.get('example_text', ''),
        height=200,
        placeholder="Paste/type text here",
        help="The model works best with at least a few sentences.",
        label_visibility="collapsed",
        key="text_input_area"  # Add a key to control the widget
    )
    
    # Character count
    char_count = len(text_input)
    st.caption(f"Character count: {char_count}")
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.example_text = ""
            st.session_state.text_input_area = ""  # Clear the text area widget
    
    # Process when button is clicked
    if analyze_button:
        if len(text_input.strip()) < 20:
            st.error("Please enter at least 20 characters for analysis.")
        else:
            # Clear any previous results by using a container
            results_container = st.container()
            
            with results_container:
                with st.spinner("Analyzing text..."):
                    # Get prediction
                    prediction, probabilities = predict_text(text_input, model, tokenizer)
                
                # Display results AFTER spinner completes
                st.markdown("---")
                st.header("Results")
                
                # Main prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("**AI-Generated**")
                    else:
                        st.success("**Human-Written**")
                
                with col2:
                    confidence = probabilities[prediction] * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Probability breakdown
                st.subheader("Probability Distribution")
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.metric("Human", f"{probabilities[0]*100:.1f}%")
                with prob_col2:
                    st.metric("AI", f"{probabilities[1]*100:.1f}%")
                
                # Progress bars for visual
                st.progress(float(probabilities[0]), text=f"Human: {probabilities[0]*100:.1f}%")
                st.progress(float(probabilities[1]), text=f"AI: {probabilities[1]*100:.1f}%")
                
                # LIME Explanation
                st.markdown("---")
                st.header("Explanation: Which Words Mattered?")
                
                with st.spinner("Generating explanation..."):
                    lime_exp = create_lime_explanation(text_input, model, tokenizer)
                    fig = plot_lime_explanation(lime_exp, prediction)
                
                st.pyplot(fig)
                
                st.markdown("""
                **How to read this:**
                - **Green bars**: Words that support the prediction
                - **Red bars**: Words that contradict the prediction
                - **Longer bars**: Stronger influence
                """)
                
                # Additional info
                with st.expander("View Full LIME Explanation"):
                    st.write("**Full LIME Explanation:**")
                    try:
                        exp_list = lime_exp.as_list(label=prediction)
                    except KeyError:
                        exp_list = lime_exp.as_list(label=1)
                    
                    for word, weight in exp_list:
                        emoji = "🟢" if weight > 0 else "🔴"
                        st.write(f"{emoji} `{word}`: {weight:.3f}")

def multiclass_classifier_page():
    """New 7-class classification UI"""
    st.title("AI Model Identifier (7-Class)")
    st.markdown("""
    **Identify which AI model generated the text:**
    - 🤖 Claude
    - 🤖 GPT-3.5
    - 🤖 GPT-4
    - 👤 Human
    - 🤖 LLaMA
    - 🤖 Mistral
    - 🤖 Other AI
    """)
    
    # Load multiclass model
    multiclass_model, multiclass_tokenizer, class_names = load_multiclass_model()
    
    # Text input
    text_input = st.text_area("Enter text to analyze", height=200)
    
    if st.button("Analyze", type="primary"):
        if len(text_input.strip()) < 20:
            st.error("Please enter at least 20 characters")
        else:
            with st.spinner("Analyzing..."):
                prediction, probabilities = predict_multiclass(
                    text_input, multiclass_model, multiclass_tokenizer
                )
            
            # Display results
            st.header("Results")
            predicted_class = class_names[prediction]
            confidence = probabilities[prediction] * 100
            
            st.success(f"**Predicted: {predicted_class}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show all probabilities
            st.subheader("All Class Probabilities")
            for i, class_name in enumerate(class_names):
                st.progress(float(probabilities[i]), text=f"{class_name}: {probabilities[i]*100:.1f}%")

@st.cache_resource
def load_multiclass_model():
    """Load the 7-class model"""
    model_path = 'results/multiclass_model/20251124_220856/best_model.pt'  # Update with your timestamp
    checkpoint = torch.load(model_path, map_location=device)

    
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=7
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    class_names = checkpoint['class_names']
    
    return model, tokenizer, class_names

def predict_multiclass(text, model, tokenizer):
    """Predict which of 7 classes the text belongs to"""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction, probabilities

if __name__ == "__main__":
    main()
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction, probabilities

if __name__ == "__main__":
    main()
