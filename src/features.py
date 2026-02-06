from typing import List, Dict, Any
import json
from src.llm_handler import LLMHandler
from src.vector_store import VectorStore
import warnings
warnings.filterwarnings("ignore")

class VideoLearningAssistant:
    def __init__(self, vector_db, llm_handler: LLMHandler, transcript: str, video_id: str = None):
        self.vector_db = vector_db
        self.llm_handler = llm_handler
        self.transcript = transcript
        self.video_id = video_id
        self.conversation_history = []
    
    def chat_with_video(self, question: str) -> str:
        """Chat with the video content"""
        
        # Use video-specific collection if available
        collection_name = f"video_{self.video_id}" if self.video_id else "video_transcripts"
        
        # Search for relevant context
        relevant_chunks = self.vector_db.search_similar(
            question, 
            collection_name=collection_name,
            k=3
        )
        
        # Combine context
        context_chunks = [chunk['text'] for chunk in relevant_chunks]
        context = "\n".join(context_chunks) if context_chunks else self.transcript[:1000]
        
        # Create system prompt
        system_prompt = """
        You are a helpful learning assistant that answers questions based on the video content.
        Use only the provided context from the video. If you don't know the answer, say so.
        Be concise and educational in your responses.
        """
        
        # Generate response
        response = self.llm_handler.generate_response(
            prompt=question,
            context=context,
            system_prompt=system_prompt
        )
        
        # Update conversation history
        self.conversation_history.append({
            "question": question,
            "response": response,
            "context_used": context_chunks
        })
        
        return response
    
    def generate_topic_summary(self, num_topics: int = 5) -> str:
        """Generate topic-wise summary of the video"""
        
        # Extract topics first
        topics = self.llm_handler.extract_topics(self.transcript[:2000])
        
        # Limit to requested number of topics
        topics = topics[:num_topics]
        
        summaries = []
        for topic in topics:
            # Use video-specific collection if available
            collection_name = f"video_{self.video_id}" if self.video_id else "video_transcripts"
            
            # Search for chunks related to this topic
            relevant_chunks = self.vector_db.search_similar(
                topic, 
                collection_name=collection_name,
                k=2
            )
            
            context_chunks = [chunk['text'] for chunk in relevant_chunks]
            context = "\n".join(context_chunks) if context_chunks else ""
            
            prompt = f"""
            Create a comprehensive summary about '{topic}' based on the following context:
            
            {context}
            
            Summary should be educational and well-structured.
            """
            
            summary = self.llm_handler.generate_response(
                prompt=prompt,
                system_prompt="You are an expert educational content summarizer."
            )
            
            summaries.append({
                "topic": topic,
                "summary": summary
            })
        
        # Format the output
        if not summaries:
            return "No topics could be extracted from the video content."
        
        formatted_summary = "# 📚 Topic-wise Summary\n\n"
        for i, item in enumerate(summaries, 1):
            formatted_summary += f"## {i}. {item['topic']}\n\n"
            formatted_summary += f"{item['summary']}\n\n"
            formatted_summary += "---\n\n"
        
        return formatted_summary
    
    def generate_quiz(self, num_questions: int = 5, difficulty: str = "medium") -> Dict:
        """Generate quiz based on video content"""
        
        prompt = f"""
        Generate {num_questions} quiz questions based on the video content.
        Difficulty level: {difficulty}
        
        Provide questions with 4 multiple choice options and indicate the correct answer.
        Format as JSON with the following structure:
        
        {{
            "quiz_title": "Video Content Quiz",
            "questions": [
                {{
                    "question": "Question text here",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": "Brief explanation"
                }}
            ]
        }}
        
        Focus on key concepts and important information from the video.
        """
        
        # Get relevant context for quiz generation
        collection_name = f"video_{self.video_id}" if self.video_id else "video_transcripts"
        relevant_chunks = self.vector_db.search_similar(
            "key concepts important points summary", 
            collection_name=collection_name,
            k=5
        )
        
        context_chunks = [chunk['text'] for chunk in relevant_chunks]
        context = "\n".join(context_chunks) if context_chunks else self.transcript[:2000]
        
        response = self.llm_handler.generate_response(
            prompt=prompt,
            context=context,
            system_prompt="You are an expert quiz creator for educational content. Return valid JSON only."
        )
        
        # Try to parse JSON response
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                quiz_data = json.loads(json_str)
                
                # Validate structure
                if 'questions' in quiz_data and isinstance(quiz_data['questions'], list):
                    return quiz_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response}")
        
        # Fallback: return structured text
        return {
            "quiz_title": "Video Content Quiz",
            "questions": [
                {
                    "question": "What was the main topic discussed in the video?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": "Review the video summary for the main topic."
                }
            ]
        }
    
    def get_learning_analytics(self) -> Dict:
        """Generate learning analytics"""
        
        # Basic analytics
        word_count = len(self.transcript.split())
        
        # Count chunks in collection
        collection_name = f"video_{self.video_id}" if self.video_id else "video_transcripts"
        chunk_count = self.vector_db.count_chunks(collection_name)
        
        # Estimate topics covered
        topics = self.llm_handler.extract_topics(self.transcript[:1500])
        
        # Conversation analytics
        total_chats = len(self.conversation_history)
        
        analytics = {
            "video_analytics": {
                "transcript_length_words": word_count,
                "estimated_duration_minutes": round(word_count / 150, 1),  # Assuming 150 WPM
                "topics_covered": len(topics),
                "main_topics": topics[:8]
            },
            "learning_analytics": {
                "total_chat_interactions": total_chats,
                "quiz_ready": True,
                "summary_generated": True,
                "recent_questions": [chat['question'] for chat in self.conversation_history[-3:]]
            },
            "recommendations": self._generate_recommendations()
        }
        
        return analytics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate personalized learning recommendations"""
        
        recommendations = [
            "Review the topic-wise summary for key concepts",
            "Take the generated quiz to test your understanding",
            "Use the chat feature to ask specific questions",
            "Focus on areas you find challenging"
        ]
        
        # Add personalized recommendations based on conversation history
        if self.conversation_history:
            last_questions = [chat['question'].lower() for chat in self.conversation_history[-3:]]
            
            if any('explain' in q for q in last_questions):
                recommendations.append("Review fundamental concepts before advancing")
            
            if any('example' in q for q in last_questions):
                recommendations.append("Practice with more examples to reinforce learning")
        
        return recommendations