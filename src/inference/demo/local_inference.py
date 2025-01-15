#!/usr/bin/env python3

import argparse
import torch
import pandas as pd
import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from src.model.architecture import AdvancedNCF


def load_model(checkpoint_path: str, device: str = "cpu") -> AdvancedNCF:
    """Load the consolidated model checkpoint"""
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


def prepare_test_data(customers_file: str, products_file: str) -> pd.DataFrame:
    """Create test pairs from customer and product data"""
    # Load data
    customers_df = pd.read_csv(customers_file)
    products_df = pd.read_csv(products_file)
    
    # Create test pairs
    test_pairs = []
    
    for _, customer in customers_df.iterrows():
        # Get customer interactions
        interactions = json.loads(customer['recent_interactions'])['recent_interactions']
        user_id = int(customer['cardnumber']) % 8031  # Match model's num_users
        
        # Add positive examples from interactions
        for interaction in interactions[:5]:  # Use first 5 interactions
            product_id = int(interaction['product_id'].lstrip('P'), 16) % 366  # Match model's num_products
            test_pairs.append({
                'user_id': user_id,
                'product_id': product_id,
                'label': 1,
                'original_product_id': interaction['product_id']
            })
            
        # Add some negative examples
        for _, product in products_df.sample(n=5).iterrows():
            product_id = int(product['product_id'].lstrip('P'), 16) % 366
            test_pairs.append({
                'user_id': user_id,
                'product_id': product_id,
                'label': 0,
                'original_product_id': product['product_id']
            })
    
    return pd.DataFrame(test_pairs)


def prepare_batch_features(df: pd.DataFrame, device: str = "cpu") -> KeyedJaggedTensor:
    """Create a KeyedJaggedTensor for a batch of examples"""
    # Prepare values and lengths for each feature
    user_values = df['user_id'].tolist()
    product_values = df['product_id'].tolist()
    
    # Combine values in the correct order
    values = torch.tensor(user_values + product_values, dtype=torch.long, device=device)
    
    # Create lengths tensor (1 for each feature for each example)
    batch_size = len(df)
    lengths = torch.tensor([1] * batch_size + [1] * batch_size, dtype=torch.long, device=device)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=["user_id", "product_id"],
        values=values,
        lengths=lengths,
    )


def main():
    parser = argparse.ArgumentParser(description="Local Inference with test data")
    parser.add_argument("--customer_csv", type=str, required=True,
                       help="Path to customer_features_enriched_sample.csv")
    parser.add_argument("--product_csv", type=str, required=True,
                       help="Path to product_features_enriched_sample.csv")
    parser.add_argument("--checkpoint", type=str, default="my_model.pt",
                       help="Path to consolidated checkpoint")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cpu", 
                       choices=["cpu", "cuda"])
    parser.add_argument("--output", type=str, default="predictions.csv",
                       help="Path to save predictions")
    args = parser.parse_args()

    # Create test DataFrame
    df = prepare_test_data(args.customer_csv, args.product_csv)
    print(f"Created {len(df)} test examples")

    # Load model
    model = load_model(args.checkpoint, device=args.device)
    print(f"Model loaded from {args.checkpoint} on {args.device}")

    # Run inference in batches
    predictions = []
    for i in range(0, len(df), args.batch_size):
        batch_df = df.iloc[i:i + args.batch_size]
        kjt = prepare_batch_features(batch_df, device=args.device)
        
        with torch.no_grad():
            scores = model(kjt)
        
        predictions.extend(scores.cpu().numpy().flatten())
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} examples...")

    # Save predictions
    df['prediction'] = predictions
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
