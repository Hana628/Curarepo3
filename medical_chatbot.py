import json
import re
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatbot:
    def __init__(self, intents_path='intents.json'):
        try:
            # Load intents from the JSON file
            with open(intents_path, 'r', encoding='utf-8') as file:
                self.intents = json.load(file)
            logger.info(f"Loaded {len(self.intents['intents'])} intents from {intents_path}")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Error loading intents file: {e}")
            self.intents = {'intents': []}
            self.is_initialized = False
        
        # Create patterns dictionary for faster lookup
        self.pattern_to_tag = {}
        for intent in self.intents['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                # Normalize pattern (lowercase, remove punctuation)
                normalized = self._normalize_text(pattern)
                self.pattern_to_tag[normalized] = tag
    
    def _normalize_text(self, text):
        """Normalize text by converting to lowercase and removing punctuation"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _get_bag_of_words(self, text):
        """Get bag of words representation of text"""
        # Normalize and split into words
        normalized = self._normalize_text(text)
        words = normalized.split()
        return set(words)
    
    def predict_intent(self, message):
        """Predict the intent based on the user's message"""
        if not self.is_initialized:
            return 'error'
        
        # Check for exact matches first
        normalized = self._normalize_text(message)
        if normalized in self.pattern_to_tag:
            return self.pattern_to_tag[normalized]
        
        # If no exact match, use word-based similarity
        message_words = self._get_bag_of_words(message)
        
        # Find the best matching intent
        best_match = None
        best_score = 0
        
        for intent in self.intents['intents']:
            tag = intent['tag']
            tag_score = 0
            
            for pattern in intent['patterns']:
                pattern_words = self._get_bag_of_words(pattern)
                
                # Count common words
                common_words = message_words.intersection(pattern_words)
                
                # Calculate similarity score
                if not common_words:
                    continue
                    
                # Calculate the proportion of matching words
                pattern_match = len(common_words) / len(pattern_words) if pattern_words else 0
                message_match = len(common_words) / len(message_words) if message_words else 0
                
                # Combined score with bonus for higher overlap
                score = (pattern_match + message_match) / 2
                
                # Update tag score with best pattern match
                tag_score = max(tag_score, score)
            
            # Update best match if this tag has a better score
            if tag_score > best_score:
                best_score = tag_score
                best_match = tag
        
        # Return best match if above threshold, or default
        if best_score >= 0.2:  # Threshold for accepting a match
            return best_match
        else:
            return 'unknown'
    
    def get_response(self, message):
        """Generate a response based on the predicted intent"""
        if not self.is_initialized:
            return {
                "response": "I'm not fully set up yet. Please check my configuration.",
                "context": ""
            }
        
        # Predict the intent
        tag = self.predict_intent(message)
        
        # Find the intent that matches the tag
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                # Get a random response from the intent
                response = random.choice(intent['responses'])
                # Get the context if available
                context = intent.get('context', [""])[0]
                return {"response": response, "context": context}
        
        # Default response if no intent matched
        return {
            "response": "I'm not sure I understand. Can you rephrase that or ask about something like blood pressure, diabetes, lifestyle recommendations, or other health concerns?",
            "context": ""
        }
    
    def get_welcome_message(self):
        """Return a welcome message"""
        if not self.is_initialized:
            return "Hello! I'm your health assistant, but I'm still getting set up."
        
        # Look for greeting intent
        for intent in self.intents['intents']:
            if intent['tag'] == 'greeting':
                return random.choice(intent['responses'])
        
        # Fallback welcome message
        return "Hello! I'm your CURA health assistant. How can I help you today?"
    
    def get_suggestion_chips(self):
        """Generate suggestion chips for chat interface"""
        # Get suggestions from the intents file
        suggestions = [
            "What can you do?",
            "Blood pressure check",
            "Diabetes risk",
            "Lifestyle advice",
            "I have a headache",
            "Flu symptoms"
        ]
        return suggestions

# Initialize the chatbot
chatbot = MedicalChatbot()

# Test the chatbot
def test_chatbot():
    print(f"Chatbot welcome: {chatbot.get_welcome_message()}")
    
    test_messages = [
        "Hello",
        "What can you do?",
        "Tell me about blood pressure",
        "I think I have diabetes",
        "I need lifestyle advice",
        "My head hurts",
        "What are symptoms of the flu?",
        "I've been coughing a lot"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = chatbot.get_response(message)
        print(f"Bot: {response['response']}")
        if response['context']:
            print(f"Context: {response['context']}")

if __name__ == "__main__":
    test_chatbot()
