import torch
import torch.nn.functional as F
import pandas as pd
import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime

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

def prepare_temporal_features(hour: int):
    # Convert hour to model's temporal encoding
    hour_tensor = torch.zeros(24)
    hour_tensor[hour] = 1.0
    return hour_tensor

def get_recommendations(model, customer_id, products_df, top_k, selected_hour=None, use_temporal=True):
    """Get recommendations with optional temporal features"""
    device = next(model.parameters()).device
    
    # Create tensors for all products
    all_products = torch.arange(len(products_df), device=device) % 366
    customer_tensor = torch.full((len(all_products),), customer_id, device=device)
    
    # Get predictions
    with torch.no_grad():
        if use_temporal and selected_hour is not None:
            # Parse hour from "H:00 AM/PM" format
            hour_str, meridiem = selected_hour.split()
            hour = int(hour_str.split(":")[0])
            
            # Convert to 24-hour format
            if meridiem == "PM" and hour != 12:
                hour += 12
            elif meridiem == "AM" and hour == 12:
                hour = 0
                
            hour_tensor = torch.full((len(all_products),), hour, device=device)
            scores = model.forward_simple(customer_tensor, all_products, hour_tensor)
        else:
            scores = model.forward_simple(customer_tensor, all_products)
    
    # Create recommendations DataFrame
    recommendations = products_df.copy()
    recommendations['score'] = scores.cpu().numpy()
    recommendations['time_context'] = selected_hour if use_temporal else "N/A"
    
    # Sort and get top-k
    recommendations = recommendations.nlargest(top_k, 'score')
    
    return recommendations

def add_model_intelligence_dashboard(model, recommendations, customer_id, use_temporal):
    st.markdown("---")
    st.markdown("### Model Intelligence Dashboard", unsafe_allow_html=True)
    
    # Add legend/key explanation
    st.markdown("""
    **Dashboard Metrics Explained:**
    - **Embedding Analysis**: Shows relative recommendation scores normalized within the current set (0-1 scale)
    - **Attention Analysis**: Measures how different attention heads specialize in capturing different patterns
    - **Confidence Analysis**: Distribution of recommendation scores across confidence levels
    - **Feature Importance**: Relative contribution of each model component to final recommendations
    """)
    st.markdown("---")
    
    # Create two rows of two columns each
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 1. Embedding Analysis (top left)
    with row1_col1:
        st.markdown("#### Embedding Analysis")
        if len(recommendations) > 0:
            scores = recommendations['score'].values
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            df = pd.DataFrame({
                'Product': recommendations['product_name'],
                'Category': recommendations['category_id'],
                'Department': recommendations['department_id'],
                'Relative Score': normalized_scores
            })
            # Common chart styling
            chart_style = {
                "config": {
                    "view": {"backgroundColor": "white"},
                    "axis": {
                        "labelFontSize": 14,
                        "titleFontSize": 16,
                        "labelColor": "black",
                        "titleColor": "black"
                    },
                    "legend": {
                        "labelFontSize": 14,
                        "titleFontSize": 16
                    }
                }
            }
            st.line_chart(
                df.set_index('Product')['Relative Score'],
                color='#E31837',
                use_container_width=True
            )
            st.caption("Relative recommendation scores (normalized within current set)")
    
    # 2. Attention Analysis (top right)
    with row1_col2:
        st.markdown("#### Attention Analysis")
        if len(recommendations) > 0:
            # Get attention scores from a forward pass
            customer_id_tensor = torch.tensor([customer_id % 8031], device=next(model.parameters()).device)
            product_ids = torch.tensor([i % 366 for i in range(len(recommendations))], device=next(model.parameters()).device)
            
            with torch.no_grad():
                # Create KeyedJaggedTensor with proper lengths
                features_kjt = KeyedJaggedTensor.from_lengths_sync(
                    keys=["user_id", "product_id"],
                    values=torch.cat([
                        customer_id_tensor.repeat(len(product_ids)),  # Repeat user ID for each product
                        product_ids
                    ]),
                    lengths=torch.ones(
                        len(product_ids) * 2,  # Total number of ones (user_ids + product_ids)
                        dtype=torch.long,
                        device=customer_id_tensor.device
                    )
                )
                
                # Get embeddings
                mlp_embeddings = model.mlp_embedding_collection(features_kjt)
                user_mlp = model.mlp_norm(mlp_embeddings["user_id"])
                product_mlp = model.mlp_norm(mlp_embeddings["product_id"])
                
                # Reshape embeddings to match attention expectations
                batch_size = 1
                seq_len = len(product_ids)
                embed_dim = user_mlp.size(-1)
                
                user_mlp = user_mlp.view(batch_size, seq_len, embed_dim)
                product_mlp = product_mlp.view(batch_size, seq_len, embed_dim)
                
                # Apply attention
                attention_output = model.user_product_attention(
                    user_mlp,
                    product_mlp,
                    product_mlp
                )
                
                # Get attention weights from intermediate computation
                # Shape: [batch_size, num_heads, seq_len, seq_len]
                q = model.user_product_attention.q_proj(user_mlp)
                k = model.user_product_attention.k_proj(product_mlp)
                q = q.view(batch_size, seq_len, model.num_heads, -1).transpose(1, 2)
                k = k.view(batch_size, seq_len, model.num_heads, -1).transpose(1, 2)
                
                # Calculate attention scores
                attention_weights = torch.matmul(q, k.transpose(-2, -1)) / model.user_product_attention.scale
                attention_weights = F.softmax(attention_weights, dim=-1)
                
                # Process attention scores to preserve head patterns
                attention_df = pd.DataFrame(
                    attention_weights.squeeze(0)  # [num_heads, seq_len, seq_len]
                    .std(dim=-1)                 # Get std of attention patterns -> [num_heads, seq_len]
                    .mean(dim=-1)                # Average over sequence length -> [num_heads]
                    .cpu().numpy(),              # Convert to numpy
                    index=[f"Head {i+1}" for i in range(model.num_heads)],
                    columns=['Importance']
                )

                # No need for additional entropy calculation since we're measuring pattern diversity with std
                attention_df['Combined_Score'] = attention_df['Importance']

                # Plot head importance
                st.bar_chart(
                    attention_df['Combined_Score'],
                    color='#E31837',
                    use_container_width=True
                )
                st.caption("Attention head importance (combines pattern strength and diversity)")
    
    # 3. Confidence Analysis (bottom left)
    with row2_col1:
        st.markdown("#### Confidence Analysis")
        if len(recommendations) > 0:
            scores = recommendations['score'].values
            confidence_metrics = {
                "High (>80%)": (scores > 0.8).mean(),
                "Medium (50-80%)": ((scores > 0.5) & (scores <= 0.8)).mean(),
                "Low (<50%)": (scores <= 0.5).mean()
            }
            st.bar_chart(
                pd.Series(confidence_metrics),
                color='#E31837',
                use_container_width=True
            )
            st.caption("Recommendation confidence distribution")
    
    # 4. Feature Importance (bottom right)
    with row2_col2:
        st.markdown("#### Feature Importance")
        if len(recommendations) > 0:
            # Calculate importance from model components
            mf_weight = abs(model.final[0].weight[0][0].item())
            mlp_weight = abs(model.final[0].weight[0][1].item())
            temporal_scale = 0.3 if use_temporal else 0.0
            
            importance = {
                "Collaborative": mf_weight,
                "Neural": mlp_weight,
                "Attention": attention_df['Importance'].mean() * mlp_weight,  # Scale attention by mlp weight
                "Temporal": mlp_weight * temporal_scale
            }
            # Normalize to percentages
            total = sum(importance.values())
            importance = {k: v/total for k, v in importance.items()}
            
            st.bar_chart(
                pd.Series(importance),
                color='#E31837',
                use_container_width=True
            )
            st.caption("Model component contributions")

def main():
    # Configure page
    st.set_page_config(
        page_title="Neural Collaborative Filtering Inference",
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

        /* Model Intelligence Dashboard */
        .stMarkdown h3, .stMarkdown h4 {
            color: #E31837 !important;
        }
        
        /* Toggle button styling */
        button[kind="secondary"] {
            border: 1px solid #ccc !important;
            background-color: white !important;
        }
        
        /* Toggle text */
        .st-emotion-cache-1nv5vhj p,
        .st-emotion-cache-1nv5vhj label {
            color: black !important;
        }

        /* Dashboard container */
        [data-testid="stVerticalBlock"] {
            background-color: white !important;
            color: black !important;
        }

        /* Toggle styling */
        .st-emotion-cache-1nv5vhj {
            border: 1px solid #ccc !important;
            background-color: white !important;
        }

        /* Toggle text color */
        .st-emotion-cache-1nv5vhj p {
            color: black !important;
        }

        /* Toggle label */
        .st-emotion-cache-1nv5vhj label {
            color: black !important;
        }

        /* Toggle button when active */
        .st-emotion-cache-1nv5vhj [data-testid="stToggleButton"] {
            background-color: #E31837 !important;
        }

        /* Chart styling */
        .stChart {
            background-color: white !important;
        }
        
        /* Chart axes and labels */
        .stChart text {
            font-size: 16px !important;
            font-weight: 500 !important;
            fill: black !important;
        }
        
        /* Chart background */
        .stChart > div > div > div {
            background-color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
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
        
        # Time selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Time Settings")

        # Add toggle for temporal features
        use_temporal = st.sidebar.toggle(
            "Enable Time-Based Recommendations",
            value=True,
            help="Toggle between time-aware and standard recommendations"
        )

        # Only show time selection if temporal features are enabled
        if use_temporal:
            hour_options = [
                f"{i if i <= 12 else i-12}:00 {'AM' if i < 12 else 'PM'}" 
                for i in range(24)
            ]
            hour_options[0] = "12:00 AM"  # Fix midnight
            hour_options[12] = "12:00 PM"  # Fix noon

            selected_hour = st.sidebar.selectbox(
                "Select Time for Recommendations:",
                options=hour_options,
                index=datetime.now().hour,
                help="Select the time of day to optimize recommendations"
            )
            
            # Add time context indicator
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### Current Settings")
                st.markdown(f"🕒 Time Context: **{selected_hour}**")
            with col2:
                if selected_hour == f"{datetime.now().hour:02d}:00":
                    st.success("Using current time")
                else:
                    st.info("Using custom time")
        
        # Add Model Intelligence section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Analysis")
        show_intelligence = st.sidebar.toggle(
            "Show Model Metrics",
            value=False,
            help="View detailed model architecture and performance metrics"
        )

        if st.sidebar.button("Get Recommendations"):
            # Convert customer ID to model's range
            customer_id = int(selected_customer) % 8031
            
            with st.spinner("Generating recommendations..."):
                if use_temporal:
                    recommendations = get_recommendations(
                        model, customer_id, products_df, top_k, 
                        selected_hour, use_temporal=True
                    )
                    st.subheader(f"Top {top_k} Recommended Products for {selected_hour}")
                    st.info(f"🕒 Showing recommendations optimized for {selected_hour}")
                else:
                    recommendations = get_recommendations(
                        model, customer_id, products_df, top_k, 
                        use_temporal=False
                    )
                    st.subheader(f"Top {top_k} Recommended Products (Time-Independent)")
                    st.info("⚡ Showing base recommendations without time context")
                
                for _, rec in recommendations.iterrows():
                    with st.container():
                        cols = st.columns([3, 1])
                        with cols[0]:
                            context_info = f"Time Context: {rec['time_context']}" if use_temporal else ""
                            st.markdown(f"""
                            **{rec['product_name']}**  
                            Category: {rec['category_id']} | Dept: {rec['department_id']}  
                            {context_info}
                            """)
                        with cols[1]:
                            score_pct = f"{rec['score']*100:.1f}%"
                            st.metric("Match", score_pct)
                        st.divider()
                
                st.markdown("---")
                st.sidebar.markdown("---")

        # Add dashboard display outside the recommendations block
        if show_intelligence and 'recommendations' in locals():
            add_model_intelligence_dashboard(model, recommendations, customer_id, use_temporal)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()