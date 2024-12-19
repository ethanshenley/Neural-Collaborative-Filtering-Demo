# test_vertex.py

import logging
from src.llm.vertex_client import VertexLLMClient

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize client
    logging.info("Initializing Vertex AI client...")
    client = VertexLLMClient(
        project_id="ce-demo-space",
        model_name="gemini-pro"
    )
    
    # Test connection
    logging.info("Testing connection...")
    if client.test_connection():
        logging.info("Connection successful!")
        
        # Test with a simple product generation prompt
        test_prompt = """Generate 3 realistic product names for the Made-to-Order (MTO) category at Sheetz.
        Each product should include size and customization options.
        Format: Product Name (Size) [Customization Options]"""
        
        logging.info("Testing product name generation...")
        response = client.generate(test_prompt)
        
        if response:
            logging.info("Generated names:")
            print(response)
        else:
            logging.error("Failed to generate product names")
    else:
        logging.error("Failed to connect to Vertex AI")

if __name__ == "__main__":
    main()