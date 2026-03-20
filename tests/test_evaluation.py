"""
Unit tests for evaluation module.
"""

import unittest
import sys
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (
    RetrievalMetrics, GenerationMetrics, RAGEvaluator,
    EvaluationResult
)

class TestRetrievalMetrics(unittest.TestCase):
    """Test cases for retrieval metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retrieved_docs = ["doc1", "doc3", "doc5", "doc2", "doc4"]
        self.relevant_docs = ["doc1", "doc2", "doc3"]
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        # Test k=3 (doc1, doc3 are relevant in top 3)
        hit_rate = RetrievalMetrics.hit_rate(self.retrieved_docs, self.relevant_docs, k=3)
        self.assertAlmostEqual(hit_rate, 2/3, places=2)
        
        # Test k=5 (doc1, doc3, doc2 are relevant in top 5)
        hit_rate = RetrievalMetrics.hit_rate(self.retrieved_docs, self.relevant_docs, k=5)
        self.assertAlmostEqual(hit_rate, 1.0, places=2)
        
        # Test with no k (all retrieved docs)
        hit_rate = RetrievalMetrics.hit_rate(self.retrieved_docs, self.relevant_docs)
        self.assertAlmostEqual(hit_rate, 1.0, places=2)
    
    def test_mean_reciprocal_rank(self):
        """Test Mean Reciprocal Rank calculation."""
        # First relevant doc is at position 0
        mrr = RetrievalMetrics.mean_reciprocal_rank(self.retrieved_docs, self.relevant_docs)
        self.assertAlmostEqual(mrr, 1.0, places=2)
        
        # Test with different retrieved order
        retrieved_alt = ["doc5", "doc1", "doc3", "doc2", "doc4"]
        mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_alt, self.relevant_docs)
        self.assertAlmostEqual(mrr, 0.5, places=2)  # 1/2
        
        # Test with no relevant docs
        mrr = RetrievalMetrics.mean_reciprocal_rank(["doc5", "doc6"], self.relevant_docs)
        self.assertAlmostEqual(mrr, 0.0, places=2)
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        # Test k=3 (doc1, doc3 are relevant out of 3)
        precision = RetrievalMetrics.precision_at_k(self.retrieved_docs, self.relevant_docs, k=3)
        self.assertAlmostEqual(precision, 2/3, places=2)
        
        # Test k=5 (doc1, doc3, doc2 are relevant out of 5)
        precision = RetrievalMetrics.precision_at_k(self.retrieved_docs, self.relevant_docs, k=5)
        self.assertAlmostEqual(precision, 3/5, places=2)
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        # Test without relevance scores (binary relevance)
        ndcg = RetrievalMetrics.ndcg_at_k(self.retrieved_docs, self.relevant_docs, k=3)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)
        
        # Test with relevance scores
        relevance_scores = {"doc1": 1.0, "doc2": 0.8, "doc3": 0.6}
        ndcg = RetrievalMetrics.ndcg_at_k(
            self.retrieved_docs, self.relevant_docs, relevance_scores, k=3
        )
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

class TestGenerationMetrics(unittest.TestCase):
    """Test cases for generation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.question = "What is the capital of France?"
        self.answer = "The capital of France is Paris."
        self.context = "Paris is the capital city of France. It is located in the north-central part of the country."
        self.retrieved_docs = [
            {"content": "Paris is the capital of France.", "id": "doc1"},
            {"content": "France is a country in Europe.", "id": "doc2"}
        ]
    
    def test_simple_faithfulness(self):
        """Test simple faithfulness calculation."""
        faithfulness = GenerationMetrics._simple_faithfulness(self.answer, self.context)
        self.assertGreaterEqual(faithfulness, 0.0)
        self.assertLessEqual(faithfulness, 1.0)
        
        # Test with completely unrelated answer
        unrelated_answer = "The sky is blue and made of candy."
        faithfulness = GenerationMetrics._simple_faithfulness(unrelated_answer, self.context)
        self.assertLess(faithfulness, 0.5)
    
    def test_simple_relevancy(self):
        """Test simple relevancy calculation."""
        relevancy = GenerationMetrics._simple_relevancy(self.answer, self.question)
        self.assertGreaterEqual(relevancy, 0.0)
        self.assertLessEqual(relevancy, 1.0)
        
        # Test with completely irrelevant answer
        irrelevant_answer = "I like pizza and ice cream."
        relevancy = GenerationMetrics._simple_relevancy(irrelevant_answer, self.question)
        self.assertLess(relevancy, 0.5)
    
    def test_simple_context_precision(self):
        """Test simple context precision calculation."""
        precision = GenerationMetrics._simple_context_precision(self.retrieved_docs, self.question)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        
        # Test with empty retrieved docs
        precision = GenerationMetrics._simple_context_precision([], self.question)
        self.assertEqual(precision, 0.0)
    
    def test_simple_context_recall(self):
        """Test simple context recall calculation."""
        recall = GenerationMetrics._simple_context_recall(self.answer, self.context)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        
        # Test with empty answer
        recall = GenerationMetrics._simple_context_recall("", self.context)
        self.assertEqual(recall, 0.0)
    
    def test_llm_based_metrics(self):
        """Test LLM-based metrics with mock."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "0.8"
        
        faithfulness = GenerationMetrics.faithfulness(self.answer, self.context, mock_llm)
        self.assertAlmostEqual(faithfulness, 0.8, places=2)
        
        relevancy = GenerationMetrics.answer_relevancy(self.answer, self.question, mock_llm)
        self.assertAlmostEqual(relevancy, 0.8, places=2)
        
        precision = GenerationMetrics.context_precision(self.retrieved_docs, self.question, mock_llm)
        self.assertAlmostEqual(precision, 0.8, places=2)
        
        recall = GenerationMetrics.context_recall(self.answer, self.context, mock_llm)
        self.assertAlmostEqual(recall, 0.8, places=2)

class TestRAGEvaluator(unittest.TestCase):
    """Test cases for RAG evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.evaluator = RAGEvaluator(self.mock_llm)
        
        self.question = "What is machine learning?"
        self.answer = "Machine learning is a subset of artificial intelligence."
        self.context = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        self.retrieved_docs = [
            {"content": "Machine learning enables systems to learn.", "id": "doc1"},
            {"content": "AI is a broad field of computer science.", "id": "doc2"}
        ]
        self.relevant_docs = ["doc1", "doc3"]
    
    def test_evaluate_retrieval(self):
        """Test retrieval evaluation."""
        retrieved_ids = ["doc1", "doc2", "doc4", "doc5"]
        
        results = self.evaluator.evaluate_retrieval(
            retrieved_ids, self.relevant_docs, k_values=[1, 3, 5]
        )
        
        # Check that all expected metrics are present
        expected_metrics = ["hit_rate@1", "hit_rate@3", "hit_rate@5", 
                          "precision@1", "precision@3", "precision@5",
                          "mrr", "ndcg@1", "ndcg@3", "ndcg@5"]
        
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)
    
    def test_evaluate_generation(self):
        """Test generation evaluation."""
        results = self.evaluator.evaluate_generation(
            self.answer, self.question, self.context, self.retrieved_docs
        )
        
        # Check that all expected metrics are present
        expected_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)
    
    def test_evaluate_rag_pipeline(self):
        """Test complete RAG pipeline evaluation."""
        results = self.evaluator.evaluate_rag_pipeline(
            self.question, self.answer, self.context, 
            self.retrieved_docs, self.relevant_docs
        )
        
        # Should include both retrieval and generation metrics
        self.assertIn("faithfulness", results)
        self.assertIn("answer_relevancy", results)
        self.assertIn("hit_rate@1", results)
        self.assertIn("mrr", results)
    
    def test_evaluate_dataset(self):
        """Test dataset evaluation."""
        dataset = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence.",
                "context": "Artificial intelligence (AI) is a broad field.",
                "retrieved_docs": [{"content": "AI is artificial intelligence.", "id": "doc1"}],
                "relevant_docs": ["doc1"]
            },
            {
                "question": "What is ML?",
                "answer": "ML is machine learning.",
                "context": "Machine learning (ML) is a subset of AI.",
                "retrieved_docs": [{"content": "ML is machine learning.", "id": "doc2"}],
                "relevant_docs": ["doc2"]
            }
        ]
        
        results = self.evaluator.evaluate_dataset(dataset)
        
        # Should include aggregated metrics
        self.assertIn("avg_faithfulness", results)
        self.assertIn("avg_answer_relevancy", results)
        self.assertIn("avg_hit_rate@1", results)
        self.assertIn("num_examples", results)
        self.assertEqual(results["num_examples"], 2)
        
        # Should include standard deviations
        self.assertIn("std_faithfulness", results)
        self.assertIn("std_answer_relevancy", results)

class TestEvaluationResult(unittest.TestCase):
    """Test cases for EvaluationResult."""
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult creation."""
        result = EvaluationResult(
            metric_name="test_metric",
            score=0.85,
            details={"key": "value"}
        )
        
        self.assertEqual(result.metric_name, "test_metric")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.details["key"], "value")
    
    def test_evaluation_result_without_details(self):
        """Test EvaluationResult without details."""
        result = EvaluationResult(
            metric_name="test_metric",
            score=0.75
        )
        
        self.assertEqual(result.metric_name, "test_metric")
        self.assertEqual(result.score, 0.75)
        self.assertIsNone(result.details)

if __name__ == "__main__":
    unittest.main()
