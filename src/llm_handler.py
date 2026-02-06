from openai import OpenAI
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore")

class LLMHandler:
    def __init__(self, model_name="gpt-3.5-turbo-16k", temperature=0.3):
        """Initialize LLM handler"""
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.langchain_llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=api_key
            )
            self.provider = "openai"
        else:
            print("OpenAI API key not found. Using HuggingFace or local model.")
            self.provider = "huggingface"
    
    def generate_response(self, prompt: str, context: str = None, 
                         system_prompt: str = None) -> str:
        """Generate response using LLM"""
        
        if self.provider == "openai":
            return self._generate_openai_response(prompt, context, system_prompt)
        else:
            return self._generate_huggingface_response(prompt, context, system_prompt)
    
    def _generate_openai_response(self, prompt: str, context: str = None,
                                 system_prompt: str = None) -> str:
        """Generate response using OpenAI API"""
        
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Context from video: {context}"
            })
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_huggingface_response(self, prompt: str, context: str = None,
                                      system_prompt: str = None) -> str:
        """Generate response using HuggingFace models"""
        # Implementation for open-source models
        # You can integrate models like Llama 2, Mistral, etc.
        
        # Placeholder - implement based on your chosen model
        return "HuggingFace model response - implement based on your model choice"
    
    def generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate summary of text"""
        prompt = f"""
        Please summarize the following text concisely:
        
        {text[:3000]}  # Limit context length
        
        Summary:
        """
        
        return self.generate_response(prompt, system_prompt="You are a helpful assistant that creates concise summaries.")
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        prompt = f"""
        Extract the main topics or concepts from the following text.
        Return them as a bullet-point list.
        
        Text: {text[:2000]}
        
        Main Topics:
        """
        
        response = self.generate_response(prompt)
        return self._parse_bullet_points(response)
    
    def _parse_bullet_points(self, text: str) -> List[str]:
        """Parse bullet points from text"""
        lines = text.split('\n')
        topics = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                # Remove bullet/number and clean
                topic = line.lstrip('-*• 1234567890.')
                if topic:
                    topics.append(topic)
        
        return topics if topics else [text]