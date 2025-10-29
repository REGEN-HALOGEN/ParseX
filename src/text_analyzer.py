"""
Text analysis module for processing extracted document text.
"""
import spacy
from typing import List, Dict, Any
from transformers import pipeline

class TextAnalyzer:
    """Handles text analysis and information extraction."""
    
    def __init__(self):
        """Initialize the text analyzer with required models."""
        self.nlp = spacy.load('en_core_web_sm')
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract key information
        analysis = {
            'entities': self._extract_entities(doc),
            'key_phrases': self._extract_key_phrases(doc),
            'summary': self._generate_summary(doc),
            'sentiment': self._analyze_sentiment(text)
        }
        
        return analysis
    
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        """Extract named entities from the text."""
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases from the text."""
        key_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                key_phrases.append(chunk.text)
        return key_phrases
    
    def _generate_summary(self, doc) -> str:
        """Generate a brief summary of the text."""
        # Simple extractive summarization
        sentences = list(doc.sents)
        if len(sentences) <= 3:
            return doc.text
            
        # Score sentences based on word importance
        word_freq = {}
        for word in doc:
            if not word.is_stop and not word.is_punct and word.text.strip():
                word_freq[word.text] = word_freq.get(word.text, 0) + 1
                
        sentence_scores = {}
        for sent in sentences:
            score = 0
            for word in sent:
                if word.text in word_freq:
                    score += word_freq[word.text]
            sentence_scores[sent] = score / len(sent)
            
        # Get top 3 sentences
        top_sentences = sorted(
            sentence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        summary = ' '.join([str(sent[0]) for sent in sorted(
            top_sentences, 
            key=lambda x: x[0].start
        )])
        
        return summary
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of the text using basic lexicon-based approach.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores
        """
        doc = self.nlp(text)
        
        # Define basic sentiment lexicons
        positive_words = {'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
                         'happy', 'pleased', 'delighted', 'satisfied', 'positive', 'beautiful'}
        negative_words = {'bad', 'poor', 'terrible', 'horrible', 'awful', 'disappointing',
                         'sad', 'angry', 'negative', 'unfortunate', 'unpleasant', 'ugly'}
        
        pos_count = 0
        neg_count = 0
        
        for token in doc:
            # Convert token to lowercase for matching
            word = token.text.lower()
            
            # Check for negation
            is_negated = any(w.dep_ == 'neg' for w in token.children)
            
            if word in positive_words:
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif word in negative_words:
                if is_negated:
                    pos_count += 1
                else:
                    neg_count += 1
        
        total = pos_count + neg_count
        if total == 0:
            return {'positive': 0.5, 'negative': 0.5}
        
        return {
            'positive': pos_count / total,
            'negative': neg_count / total
        }