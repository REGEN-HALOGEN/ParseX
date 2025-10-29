"""
Tests for the text analyzer module.
"""
import pytest
from src.text_analyzer import TextAnalyzer

@pytest.fixture
def analyzer():
    """Create a TextAnalyzer instance."""
    return TextAnalyzer()

def test_text_analyzer_initialization(analyzer):
    """Test TextAnalyzer initialization."""
    assert analyzer.nlp is not None

def test_entity_extraction(analyzer):
    """Test named entity extraction."""
    text = "Microsoft Corporation was founded by Bill Gates in Seattle."
    analysis = analyzer.analyze_text(text)
    entities = analysis['entities']
    
    assert isinstance(entities, list)
    assert len(entities) > 0
    assert all(isinstance(e, dict) for e in entities)
    assert all(set(e.keys()) == {'text', 'label', 'start', 'end'} for e in entities)

def test_key_phrase_extraction(analyzer):
    """Test key phrase extraction."""
    text = "The quick brown fox jumped over the lazy dog."
    analysis = analyzer.analyze_text(text)
    phrases = analysis['key_phrases']
    
    assert isinstance(phrases, list)
    assert len(phrases) > 0
    assert all(isinstance(p, str) for p in phrases)

def test_summary_generation(analyzer):
    """Test summary generation."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    analysis = analyzer.analyze_text(text)
    summary = analysis['summary']
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) <= len(text)

def test_sentiment_analysis(analyzer):
    """Test sentiment analysis."""
    # Test positive sentiment
    text = "This is a good and wonderful product! I'm very happy with it."
    analysis = analyzer.analyze_text(text)
    sentiment = analysis['sentiment']
    
    assert isinstance(sentiment, dict)
    assert 'positive' in sentiment
    assert 'negative' in sentiment
    assert 0 <= sentiment['positive'] <= 1
    assert 0 <= sentiment['negative'] <= 1
    assert sentiment['positive'] > sentiment['negative']
    
    # Test negative sentiment
    text = "This is a terrible and disappointing product. I'm very sad."
    analysis = analyzer.analyze_text(text)
    sentiment = analysis['sentiment']
    
    assert isinstance(sentiment, dict)
    assert 'positive' in sentiment
    assert 'negative' in sentiment
    assert 0 <= sentiment['positive'] <= 1
    assert 0 <= sentiment['negative'] <= 1
    assert sentiment['negative'] > sentiment['positive']