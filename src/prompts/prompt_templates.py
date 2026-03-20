"""
Prompt templates for RAG system.
Contains various prompt templates for different use cases.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Represents a prompt template."""
    name: str
    template: str
    description: str
    variables: List[str]
    category: str = "general"

class PromptTemplates:
    """Collection of prompt templates for RAG applications."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates."""
        templates = {}
        
        # RAG Question Answering Templates
        templates["qa_basic"] = PromptTemplate(
            name="qa_basic",
            template="""Based on the following context, please answer the question. 
If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {question}

Answer:""",
            description="Basic question answering template",
            variables=["context", "question"],
            category="qa"
        )
        
        templates["qa_detailed"] = PromptTemplate(
            name="qa_detailed",
            template="""Please answer the following question based on the provided context. 
Provide a detailed and comprehensive answer, citing specific information from the context when possible.

Context:
{context}

Question: {question}

Instructions:
- Use only the information provided in the context
- If the context doesn't contain the answer, clearly state that
- Provide specific details and examples from the context when relevant
- Structure your answer clearly with bullet points or numbered lists when appropriate

Answer:""",
            description="Detailed question answering with citations",
            variables=["context", "question"],
            category="qa"
        )
        
        templates["qa_concise"] = PromptTemplate(
            name="qa_concise",
            template="""Answer the question using only the provided context. Be concise and direct.

Context:
{context}

Question: {question}

Answer:""",
            description="Conise question answering template",
            variables=["context", "question"],
            category="qa"
        )
        
        # Context Summarization Templates
        templates["summarize_context"] = PromptTemplate(
            name="summarize_context",
            template="""Please summarize the following context in a clear and concise manner.

Context:
{context}

Summary:""",
            description="Context summarization template",
            variables=["context"],
            category="summarization"
        )
        
        templates["extract_key_points"] = PromptTemplate(
            name="extract_key_points",
            template="""Extract the key points from the following context. Present them as a bulleted list.

Context:
{context}

Key Points:""",
            description="Extract key points from context",
            variables=["context"],
            category="summarization"
        )
        
        # Question Generation Templates
        templates["generate_questions"] = PromptTemplate(
            name="generate_questions",
            template="""Based on the following context, generate {num_questions} relevant questions that could be answered using this information.

Context:
{context}

Generated Questions:""",
            description="Generate questions from context",
            variables=["context", "num_questions"],
            category="generation"
        )
        
        templates["generate_faq"] = PromptTemplate(
            name="generate_faq",
            template"""Create an FAQ section based on the following context. Include at least {num_questions} question-answer pairs.

Context:
{context}

FAQ:""",
            description="Generate FAQ from context",
            variables=["context", "num_questions"],
            category="generation"
        )
        
        # Classification Templates
        templates["classify_sentiment"] = PromptTemplate(
            name="classify_sentiment",
            template="""Classify the sentiment of the following text as positive, negative, or neutral.

Text:
{text}

Sentiment:""",
            description="Sentiment classification template",
            variables=["text"],
            category="classification"
        )
        
        templates["classify_topic"] = PromptTemplate(
            name="classify_topic",
            template="""Classify the main topic of the following text.

Text:
{text}

Main Topic:""",
            description="Topic classification template",
            variables=["text"],
            category="classification"
        )
        
        # Comparison Templates
        templates["compare_documents"] = PromptTemplate(
            name="compare_documents",
            template="""Compare the following two documents and highlight their similarities and differences.

Document 1:
{document1}

Document 2:
{document2}

Comparison:""",
            description="Compare two documents",
            variables=["document1", "document2"],
            category="comparison"
        )
        
        # Explanation Templates
        templates["explain_concept"] = PromptTemplate(
            name="explain_concept",
            template="""Explain the concept of '{concept}' based on the following context. Make it easy to understand.

Context:
{context}

Explanation:""",
            description="Explain a concept based on context",
            variables=["concept", "context"],
            category="explanation"
        )
        
        templates["step_by_step"] = PromptTemplate(
            name="step_by_step",
            template="""Provide a step-by-step explanation for: {task}

Context:
{context}

Step-by-step explanation:""",
            description="Step-by-step explanation template",
            variables=["task", "context"],
            category="explanation"
        )
        
        # Validation Templates
        templates["validate_answer"] = PromptTemplate(
            name="validate_answer",
            template="""Review the following answer and determine if it's accurate based on the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Validation (Is this answer accurate based on the context? Yes/No with explanation):""",
            description="Validate answer accuracy",
            variables=["context", "question", "answer"],
            category="validation"
        )
        
        templates["fact_check"] = PromptTemplate(
            name="fact_check",
            template="""Fact-check the following statement using the provided context.

Statement: {statement}

Context:
{context}

Fact-check result:""",
            description="Fact-check statement against context",
            variables=["statement", "context"],
            category="validation"
        )
        
        return templates
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]
    
    def format_template(self, name: str, **kwargs) -> str:
        """Format a template with provided variables."""
        template = self.get_template(name)
        
        # Check if all required variables are provided
        missing_vars = [var for var in template.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return template.template.format(**kwargs)
    
    def list_templates(self, category: str = None) -> List[PromptTemplate]:
        """List all templates, optionally filtered by category."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return templates
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        categories = set(template.category for template in self.templates.values())
        return sorted(list(categories))
    
    def add_template(self, template: PromptTemplate):
        """Add a new template."""
        self.templates[template.name] = template
    
    def remove_template(self, name: str) -> bool:
        """Remove a template by name."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def search_templates(self, query: str) -> List[PromptTemplate]:
        """Search templates by name or description."""
        query_lower = query.lower()
        matching_templates = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower()):
                matching_templates.append(template)
        
        return matching_templates
    
    def get_template_variables(self, name: str) -> List[str]:
        """Get variables required by a template."""
        template = self.get_template(name)
        return template.variables
    
    def validate_template_variables(self, name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if all required variables are provided for a template."""
        template = self.get_template(name)
        required_vars = set(template.variables)
        provided_vars = set(variables.keys())
        
        missing = required_vars - provided_vars
        extra = provided_vars - required_vars
        
        return {
            "valid": len(missing) == 0,
            "missing": list(missing),
            "extra": list(extra)
        }
