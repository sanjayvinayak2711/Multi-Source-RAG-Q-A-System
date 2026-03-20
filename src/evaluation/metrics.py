"""
RAG evaluation metrics.
Implements various metrics for evaluating retrieval and generation quality.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import math

@dataclass
class EvaluationResult:
    """Result of an evaluation metric."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None

class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, **kwargs) -> EvaluationResult:
        """Compute the metric."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the metric name."""
        pass

class RetrievalMetrics:
    """Collection of retrieval evaluation metrics."""
    
    @staticmethod
    def hit_rate(retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        """
        Calculate Hit Rate (Recall@K).
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider (None for all)
            
        Returns:
            Hit rate score
        """
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        
        hits = len(set(retrieved_docs) & set(relevant_docs))
        return hits / len(relevant_docs) if relevant_docs else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            
        Returns:
            MRR score
        """
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Precision@K score
        """
        retrieved_k = retrieved_docs[:k]
        hits = len(set(retrieved_k) & set(relevant_docs))
        return hits / k if k > 0 else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], 
                  relevance_scores: Dict[str, float] = None, k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            relevance_scores: Optional relevance scores for documents
            k: Number of top documents to consider
            
        Returns:
            NDCG@K score
        """
        if relevance_scores is None:
            # Binary relevance: relevant docs have score 1, others 0
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = sorted([relevance_scores.get(doc_id, 0.0) 
                                  for doc_id in relevant_docs], reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0

class GenerationMetrics:
    """Collection of generation evaluation metrics."""
    
    @staticmethod
    def faithfulness(answer: str, context: str, llm_client=None) -> float:
        """
        Calculate faithfulness score (how well the answer is supported by context).
        
        Args:
            answer: Generated answer
            context: Source context
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Faithfulness score (0-1)
        """
        if llm_client is None:
            # Simple heuristic-based evaluation
            return GenerationMetrics._simple_faithfulness(answer, context)
        
        # LLM-based evaluation
        prompt = f"""
        Evaluate the faithfulness of the following answer based on the provided context.
        Rate the faithfulness on a scale of 0 to 1, where:
        - 1.0: Answer is fully supported by the context
        - 0.5: Answer is partially supported by the context
        - 0.0: Answer is not supported by the context
        
        Context: {context}
        Answer: {answer}
        
        Provide only the numerical score (0.0, 0.5, or 1.0):
        """
        
        try:
            response = llm_client.generate(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return GenerationMetrics._simple_faithfulness(answer, context)
    
    @staticmethod
    def _simple_faithfulness(answer: str, context: str) -> float:
        """Simple heuristic-based faithfulness evaluation."""
        # Extract key phrases from answer
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        # Calculate overlap
        overlap = len(answer_words & context_words)
        total_answer_words = len(answer_words)
        
        return overlap / total_answer_words if total_answer_words > 0 else 0.0
    
    @staticmethod
    def answer_relevancy(answer: str, question: str, llm_client=None) -> float:
        """
        Calculate answer relevancy (how relevant the answer is to the question).
        
        Args:
            answer: Generated answer
            question: Original question
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Answer relevancy score (0-1)
        """
        if llm_client is None:
            # Simple heuristic-based evaluation
            return GenerationMetrics._simple_relevancy(answer, question)
        
        # LLM-based evaluation
        prompt = f"""
        Evaluate the relevancy of the following answer to the question.
        Rate the relevancy on a scale of 0 to 1, where:
        - 1.0: Answer is highly relevant to the question
        - 0.5: Answer is somewhat relevant to the question
        - 0.0: Answer is not relevant to the question
        
        Question: {question}
        Answer: {answer}
        
        Provide only the numerical score (0.0, 0.5, or 1.0):
        """
        
        try:
            response = llm_client.generate(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return GenerationMetrics._simple_relevancy(answer, question)
    
    @staticmethod
    def _simple_relevancy(answer: str, question: str) -> float:
        """Simple heuristic-based relevancy evaluation."""
        # Extract key terms from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words -= stopwords
        answer_words -= stopwords
        
        # Calculate overlap
        overlap = len(question_words & answer_words)
        total_question_words = len(question_words)
        
        return overlap / total_question_words if total_question_words > 0 else 0.0
    
    @staticmethod
    def context_precision(retrieved_docs: List[Dict[str, Any]], question: str, 
                        llm_client=None) -> float:
        """
        Calculate context precision (how relevant the retrieved context is).
        
        Args:
            retrieved_docs: List of retrieved documents with content
            question: Original question
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Context precision score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        if llm_client is None:
            # Simple heuristic-based evaluation
            return GenerationMetrics._simple_context_precision(retrieved_docs, question)
        
        # LLM-based evaluation
        total_score = 0.0
        
        for doc in retrieved_docs:
            prompt = f"""
            Evaluate the relevance of the following context to the question.
            Rate the relevance on a scale of 0 to 1, where:
            - 1.0: Context is highly relevant to the question
            - 0.5: Context is somewhat relevant to the question
            - 0.0: Context is not relevant to the question
            
            Question: {question}
            Context: {doc.get('content', '')}
            
            Provide only the numerical score (0.0, 0.5, or 1.0):
            """
            
            try:
                response = llm_client.generate(prompt)
                score = float(response.strip())
                total_score += max(0.0, min(1.0, score))
            except:
                # Fallback to simple evaluation
                total_score += GenerationMetrics._simple_doc_relevance(doc.get('content', ''), question)
        
        return total_score / len(retrieved_docs)
    
    @staticmethod
    def _simple_context_precision(retrieved_docs: List[Dict[str, Any]], question: str) -> float:
        """Simple heuristic-based context precision evaluation."""
        total_score = 0.0
        
        for doc in retrieved_docs:
            score = GenerationMetrics._simple_doc_relevance(doc.get('content', ''), question)
            total_score += score
        
        return total_score / len(retrieved_docs) if retrieved_docs else 0.0
    
    @staticmethod
    def _simple_doc_relevance(doc_content: str, question: str) -> float:
        """Simple document relevance evaluation."""
        if not doc_content:
            return 0.0
        
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        doc_words = set(re.findall(r'\b\w+\b', doc_content.lower()))
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words -= stopwords
        
        # Calculate overlap
        overlap = len(question_words & doc_words)
        total_question_words = len(question_words)
        
        return overlap / total_question_words if total_question_words > 0 else 0.0
    
    @staticmethod
    def context_recall(answer: str, context: str, llm_client=None) -> float:
        """
        Calculate context recall (how much of the answer is supported by context).
        
        Args:
            answer: Generated answer
            context: Source context
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Context recall score (0-1)
        """
        if llm_client is None:
            # Simple heuristic-based evaluation
            return GenerationMetrics._simple_context_recall(answer, context)
        
        # LLM-based evaluation
        prompt = f"""
        Evaluate how much of the following answer is supported by the context.
        Rate the context recall on a scale of 0 to 1, where:
        - 1.0: All parts of the answer are supported by the context
        - 0.5: Some parts of the answer are supported by the context
        - 0.0: No parts of the answer are supported by the context
        
        Context: {context}
        Answer: {answer}
        
        Provide only the numerical score (0.0, 0.5, or 1.0):
        """
        
        try:
            response = llm_client.generate(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return GenerationMetrics._simple_context_recall(answer, context)
    
    @staticmethod
    def _simple_context_recall(answer: str, context: str) -> float:
        """Simple heuristic-based context recall evaluation."""
        # Extract sentences from answer
        answer_sentences = re.split(r'[.!?]+', answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        supported_sentences = 0
        
        for sentence in answer_sentences:
            # Check if sentence words appear in context
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            context_words = set(re.findall(r'\b\w+\b', context.lower()))
            
            overlap = len(sentence_words & context_words)
            if overlap / len(sentence_words) > 0.5:  # At least 50% overlap
                supported_sentences += 1
        
        return supported_sentences / len(answer_sentences)

class RAGEvaluator:
    """Main RAG evaluator that combines multiple metrics."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
    
    def evaluate_retrieval(self, 
                          retrieved_docs: List[str], 
                          relevant_docs: List[str],
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary of retrieval metrics
        """
        results = {}
        
        # Hit Rate at different k values
        for k in k_values:
            results[f"hit_rate@{k}"] = self.retrieval_metrics.hit_rate(
                retrieved_docs, relevant_docs, k
            )
        
        # Precision at different k values
        for k in k_values:
            results[f"precision@{k}"] = self.retrieval_metrics.precision_at_k(
                retrieved_docs, relevant_docs, k
            )
        
        # MRR
        results["mrr"] = self.retrieval_metrics.mean_reciprocal_rank(
            retrieved_docs, relevant_docs
        )
        
        # NDCG at different k values
        for k in k_values:
            results[f"ndcg@{k}"] = self.retrieval_metrics.ndcg_at_k(
                retrieved_docs, relevant_docs, k=k
            )
        
        return results
    
    def evaluate_generation(self, 
                           answer: str, 
                           question: str, 
                           context: str,
                           retrieved_docs: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate generation performance.
        
        Args:
            answer: Generated answer
            question: Original question
            context: Source context
            retrieved_docs: List of retrieved documents
            
        Returns:
            Dictionary of generation metrics
        """
        results = {}
        
        # Faithfulness
        results["faithfulness"] = self.generation_metrics.faithfulness(
            answer, context, self.llm_client
        )
        
        # Answer relevancy
        results["answer_relevancy"] = self.generation_metrics.answer_relevancy(
            answer, question, self.llm_client
        )
        
        # Context precision
        if retrieved_docs:
            results["context_precision"] = self.generation_metrics.context_precision(
                retrieved_docs, question, self.llm_client
            )
        
        # Context recall
        results["context_recall"] = self.generation_metrics.context_recall(
            answer, context, self.llm_client
        )
        
        return results
    
    def evaluate_rag_pipeline(self, 
                            question: str,
                            answer: str,
                            context: str,
                            retrieved_docs: List[Dict[str, Any]],
                            relevant_docs: List[str] = None,
                            k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate complete RAG pipeline.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Source context
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant document IDs (optional)
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary of all metrics
        """
        results = {}
        
        # Generation metrics
        generation_results = self.evaluate_generation(answer, question, context, retrieved_docs)
        results.update(generation_results)
        
        # Retrieval metrics (if relevant docs provided)
        if relevant_docs:
            retrieved_ids = [doc.get('id', '') for doc in retrieved_docs]
            retrieval_results = self.evaluate_retrieval(retrieved_ids, relevant_docs, k_values)
            results.update(retrieval_results)
        
        return results
    
    def evaluate_dataset(self, 
                        dataset: List[Dict[str, Any]],
                        k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline on a dataset.
        
        Args:
            dataset: List of evaluation examples
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with aggregated metrics
        """
        all_results = []
        
        for example in dataset:
            result = self.evaluate_rag_pipeline(
                question=example.get('question', ''),
                answer=example.get('answer', ''),
                context=example.get('context', ''),
                retrieved_docs=example.get('retrieved_docs', []),
                relevant_docs=example.get('relevant_docs'),
                k_values=k_values
            )
            all_results.append(result)
        
        # Aggregate results
        aggregated = {}
        if all_results:
            # Calculate mean for each metric
            metric_names = all_results[0].keys()
            for metric in metric_names:
                values = [result.get(metric, 0) for result in all_results if metric in result]
                if values:
                    aggregated[f"avg_{metric}"] = np.mean(values)
                    aggregated[f"std_{metric}"] = np.std(values)
        
        aggregated["num_examples"] = len(all_results)
        
        return aggregated
