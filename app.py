import streamlit as st
from PIL import Image
import tempfile
import os
import pandas as pd

from main import get_similar_images   # your similarity logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Image Similarity Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.block-container {
    background-color: #f7f9fc;
    padding: 25px;
    border-radius: 15px;
}
img {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🖼️ Image Similarity Search System")
st.write("AI-based image retrieval using **MobileNetV2 + Cosine Similarity**")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Search Settings")

top_k = st.sidebar.slider(
    "Number of similar images",
    min_value=3,
    max_value=10,
    value=5
)

st.sidebar.markdown("### 🔍 Model Details")
st.sidebar.info("""
• CNN Model: MobileNetV2  
• Feature Matching: Cosine Similarity  
• Dataset Size: 5,727 Images  
""")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload a query image",
    type=["jpg", "png", "jpeg"]
)

# ---------------- MAIN LOGIC ----------------
if uploaded_file:

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        query_path = tmp.name

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Search Results", "📊 Analysis"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("🔎 Query Image")
        st.image(Image.open(query_path), width=300)

        with st.spinner("Extracting features & searching similar images..."):
            similar_images, similarity_scores = get_similar_images(
                query_path, top_k
            )

        st.info(
            "Similar images are retrieved using deep feature extraction from "
            "MobileNetV2 and cosine similarity comparison."
        )

        # Similar Images Section
        with st.container():
            st.subheader("📸 Top Similar Images")

            cols = st.columns(len(similar_images))
            for i, col in enumerate(cols):
                with col:
                    img = Image.open(similar_images[i])
                    st.image(img, use_container_width=True)
                    st.caption(f"Similarity Score: {similarity_scores[i]:.2f}")

    # ---------------- TAB 2 ----------------
    with tab2:
        df = pd.DataFrame({
            "Image": [f"Image {i+1}" for i in range(len(similarity_scores))],
            "Similarity Score": similarity_scores
        })

        st.subheader("📊 Similarity Score Distribution")
        st.bar_chart(df.set_index("Image"))

    # Cleanup temp file
    os.remove(query_path)

    
