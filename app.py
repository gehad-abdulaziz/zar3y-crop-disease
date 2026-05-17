import streamlit as st
import requests
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Zar3y - AI Crop Assistant",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        border-color: #1b5e20;
    }
    .diagnosis-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .confidence-meter {
        height: 10px;
        border-radius: 5px;
        background-color: #e0e0e0;
        margin-top: 5px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        background-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.title("Zar3y (زرعي)")
    st.info("AI-powered assistant for Egyptian farmers to detect crop diseases instantly.")
    st.markdown("---")
    st.write("**Supported Crops:**")
    st.write("🍅 Tomato, 🥔 Potato, 🫑 Pepper, 🌽 Corn")

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.title("🌿 Zar3y Crop Health")
    st.write("Upload a leaf photo to get an instant diagnosis.")
    
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a photo")
    
    file = uploaded_file or camera_file

with col2:
    if file is not None:
        st.image(file, caption='Uploaded Leaf', use_container_width=True)
        
        if st.button('Analyze Health 🔍'):
            with st.spinner('Zar3y is analyzing...'):
                try:
                    # Prepare file for FastAPI
                    files = {"file": file.getvalue()}
                    response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.markdown("### 📊 Diagnosis Results")
                        
                        # Diagnosis Card
                        st.markdown(f"""
                        <div class="diagnosis-card">
                            <h2 style='color: #2e7d32; margin-top:0;'>{data['class_name'].replace('___', ' ')}</h2>
                            <p><strong>Confidence:</strong> {data['confidence']*100:.1f}%</p>
                            <div class="confidence-meter">
                                <div class="confidence-fill" style="width: {data['confidence']*100}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Visual Explanation (Grad-CAM)
                        if data.get('heatmap'):
                            st.write("### 🔍 Why Zar3y think so?")
                            st.write("The red areas in this heatmap show where the AI detected the disease symptoms.")
                            heatmap_image = base64.b64decode(data['heatmap'])
                            st.image(heatmap_image, caption='AI Attention Heatmap (Grad-CAM)', use_container_width=True)
                        
                        st.markdown("### 📝 Next Steps")
                        st.success(data['explanation'])
                        
                    else:
                        st.error("Error connecting to backend. Is the FastAPI server running?")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
    else:
        st.write("Waiting for image...")
        st.image("https://via.placeholder.com/400x300.png?text=Upload+Leaf+to+Start", use_container_width=True)

st.markdown("---")
st.caption("Developed for graduation project. Always consult an agronomist for critical decisions.")
