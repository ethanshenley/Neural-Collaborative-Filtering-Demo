import torch
import pandas as pd
import streamlit as st
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from src.model.architecture import AdvancedNCF

def load_model(checkpoint_path: str, device: str = "cpu") -> AdvancedNCF:
    """Load the model from checkpoint"""
    model = AdvancedNCF(
        num_users=8031,
        num_products=366,
        num_departments=5,
        num_categories=24,
        mf_embedding_dim=64,
        mlp_embedding_dim=64,
        temporal_dim=32,
        mlp_hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.2,
        negative_samples=4
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def get_recommendations(model, user_id: int, products_df: pd.DataFrame, top_k: int = 10):
    """Get top-k product recommendations for a user"""
    # Create test pairs for all products
    test_pairs = []
    for _, product in products_df.iterrows():
        product_id = int(product['product_id'].lstrip('P'), 16) % 366
        test_pairs.append({
            'user_id': user_id % 8031,  # Match model's num_users
            'product_id': product_id,
            'original_product_id': product['product_id']
        })
    
    df = pd.DataFrame(test_pairs)
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Create KeyedJaggedTensor for inference
    user_values = df['user_id'].tolist()
    product_values = df['product_id'].tolist()
    values = torch.tensor(user_values + product_values, dtype=torch.long, device=device)
    lengths = torch.tensor([1] * len(df) * 2, dtype=torch.long, device=device)
    
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["user_id", "product_id"],
        values=values,
        lengths=lengths,
    )
    
    # Get predictions
    with torch.no_grad():
        scores = torch.sigmoid(model(kjt)).cpu().numpy()
    
    # Add scores to DataFrame
    df['score'] = scores
    
    # Merge with product info
    results = df.merge(products_df, left_on='original_product_id', right_on='product_id', suffixes=('', '_full'))
    
    # Sort and get top k
    return results.nlargest(top_k, 'score')

def main():
    # Configure page
    st.set_page_config(
        page_title="Sheetz Product Recommender",
        page_icon="🛒",
        layout="wide"
    )

    # Custom CSS for Sheetz branding
    st.markdown("""
        <style>
        /* Main background and text colors */
        .stApp {
            background-color: white;
        }
        
        /* All text black by default */
        .stApp, .stMarkdown, p, span, div, .stSelectbox, .stSlider {
            color: black !important;
        }
        
        /* Style header */
        header {
            background-color: #ffffff !important;
            border-bottom: 2px solid #f0f0f0;
        }
        
        /* Sheetz red for headers and accents */
        .css-10trblm, .css-1dp5vir {
            color: #E31837 !important;
        }
        
        /* Style sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 2px solid #f0f0f0;
        }
        
        /* Style buttons */
        .stButton>button {
            background-color: #E31837;
            color: white !important;
        }
        
        /* Style metrics */
        .css-1xarl3l {
            background-color: #E31837;
            color: white !important;
        }
        
        /* Force black text in specific components */
        .stSlider label {
            color: black !important;
        }
        
        /* Style selectbox label */
        .stSelectbox label {
            color: #E31837 !important;
        }
        
        /* Product details text */
        .element-container div {
            color: black !important;
        }
        
        /* Sidebar header */
        .css-1adrfps {
            background-color: #ffffff !important;
        }
        
        /* Style dropdown/selectbox */
        div[data-baseweb="select"] {
            background-color: white !important;
        }

        /* Style dropdown input container */
        div[data-baseweb="select"] > div {
            background-color: white !important;
        }

        div[data-baseweb="select"] div[data-testid="stSelectbox"] {
            background-color: white !important;
        }

        /* Style dropdown menu container */
        div[data-baseweb="popover"] {
            background-color: white !important;
        }

        /* Style dropdown menu and options */
        div[data-baseweb="popover"] > div {
            background-color: white !important;
        }

        div[data-baseweb="popover"] div[role="listbox"] {
            background-color: white !important;
        }

        div[data-baseweb="popover"] div[role="option"] {
            background-color: white !important;
            color: black !important;
        }

        /* Style hover and selected states */
        div[data-baseweb="popover"] div[role="option"]:hover {
            background-color: #fff5f5 !important;
        }

        div[data-baseweb="popover"] div[role="option"][aria-selected="true"] {
            background-color: #ffe5e5 !important;
        }

        /* Force white background on all popover children */
        div[data-baseweb="popover"] * {
            background-color: white !important;
        }

        /* Logo and title alignment */
        div[data-testid="column"] img {
            margin-top: 1.2rem !important;
        }
        
        div[data-testid="column"] h1 {
            margin-top: 1.2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add Sheetz logo
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("src/inference/demo/sheetz.png", width=200)
    with col2:
        st.markdown('<h1 style="margin-top: 1.8rem;">Sheetz Product Recommender</h1>', unsafe_allow_html=True)
    
    try:
        # Load data with correct paths
        data_dir = Path(__file__).parent / "data"
        customers_df = pd.read_csv(data_dir / "customer_features_enriched_sample.csv")
        products_df = pd.read_csv(data_dir / "product_features_enriched_sample.csv")
        
        # Load model with correct path
        model_path = Path(__file__).parent / "my_model.pt"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(model_path, device)
        st.success("✅ Model and data loaded successfully!")
        
        # User input
        st.sidebar.header("Settings")
        selected_customer = st.sidebar.selectbox(
            "Select Customer:",
            options=customers_df['cardnumber'].tolist(),
            format_func=lambda x: f"Customer {str(x)[:8]}..."
        )
        top_k = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=20, value=10)
        
        if st.sidebar.button("Get Recommendations"):
            # Convert customer ID to model's range
            customer_id = int(selected_customer) % 8031
            
            with st.spinner("Generating recommendations..."):
                recommendations = get_recommendations(model, customer_id, products_df, top_k)
                
                st.subheader(f"Top {top_k} Recommended Products")
                for _, rec in recommendations.iterrows():
                    with st.container():
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"""
                            **{rec['product_name']}**  
                            Category: {rec['category_id']} | Dept: {rec['department_id']}
                            """)
                        with cols[1]:
                            score_pct = f"{rec['score']*100:.1f}%"
                            st.metric("Match", score_pct)
                        st.divider()
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()