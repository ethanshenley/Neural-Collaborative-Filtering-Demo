# verify_setup.py
import torch
import torchrec
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

def test_torchrec_setup():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"TorchRec version: {torchrec.__version__}")
    
    # Create a simple embedding bag collection
    ebc = torchrec.EmbeddingBagCollection(
        device="cpu",  # Start with CPU to verify basic functionality
        tables=[
            torchrec.EmbeddingBagConfig(
                name="test_table",
                embedding_dim=16,
                num_embeddings=100,
                feature_names=["test"],
                pooling=torchrec.PoolingType.SUM,
            )
        ]
    )
    
    # Create test data
    test_jt = JaggedTensor(
        values=torch.tensor([1, 2, 3]), 
        lengths=torch.tensor([2, 1])
    )
    
    kjt = KeyedJaggedTensor.from_jt_dict({"test": test_jt})
    
    # Try forward pass
    try:
        output = ebc(kjt)
        print("Forward pass successful!")
        print(f"Output shape: {output['test'].shape}")
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")

if __name__ == "__main__":
    test_torchrec_setup()