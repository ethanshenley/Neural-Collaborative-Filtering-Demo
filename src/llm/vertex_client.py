# src/llm/vertex_client.py

import logging
from typing import Optional, Dict, List
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel

class VertexLLMClient:
    """Client for interacting with Vertex AI LLMs"""
    
    def __init__(self, 
                 project_id: str = "ce-demo-space",
                 location: str = "us-central1",
                 model_name: str = "gemini-pro"):
        """Initialize Vertex AI client
        
        Args:
            project_id: GCP project ID
            location: GCP region
            model_name: Either 'gemini-pro' or 'text-bison@002'
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location,
        )
        
        # Initialize model based on selection
        if model_name == "gemini-pro":
            self.model = GenerativeModel("gemini-pro")
        else:
            self.model = TextGenerationModel.from_pretrained("text-bison@002")
            
        logging.info(f"Initialized Vertex AI client with {model_name}")
        
    def generate(self, 
                prompt: str,
                temperature: float = 0.7,
                max_retries: int = 3) -> Optional[str]:
        """Generate text using Vertex AI
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_retries: Maximum number of retries on failure
            
        Returns:
            Generated text or None if all retries fail
        """
        for attempt in range(max_retries):
            try:
                if self.model_name == "gemini-pro":
                    response = self.model.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    return response.text
                else:
                    response = self.model.predict(
                        prompt,
                        temperature=temperature
                    )
                    return response.text
                    
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logging.error("All retries failed")
                    return None
                    
    def generate_batch(self, 
                      prompts: List[str],
                      temperature: float = 0.7,
                      max_retries: int = 3) -> List[Optional[str]]:
        """Generate text for multiple prompts
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature (0.0 to 1.0)
            max_retries: Maximum number of retries on failure
            
        Returns:
            List of generated texts (None for failed generations)
        """
        return [
            self.generate(prompt, temperature, max_retries)
            for prompt in prompts
        ]
        
    def test_connection(self) -> bool:
        """Test the connection to Vertex AI"""
        try:
            test_prompt = "Generate a test response."
            response = self.generate(test_prompt)
            return response is not None
        except Exception as e:
            logging.error(f"Connection test failed: {str(e)}")
            return False