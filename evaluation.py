"""
RAGAS Evaluation Framework
Implements evaluation metrics for RAG system using RAGAS
"""

import os
import json
import openai
import pandas as pd
from typing import List, Dict, Any
# Force reload ragas if it's in sys.modules
import sys
if 'ragas' in sys.modules:
    del sys.modules['ragas']

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        ContextRelevance
    )
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    # Create dummy functions
    def evaluate(*args, **kwargs):
        return {'error': 'RAGAS not available'}
    
    # Log the import error
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import RAGAS: {e}")
    faithfulness = answer_relevancy = context_precision = context_recall = ContextRelevance = None

# Force reload datasets if it's in sys.modules
if 'datasets' in sys.modules:
    del sys.modules['datasets']

try:
    from datasets import Dataset
except ImportError as e:
    Dataset = None
    # Log the import error
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import datasets: {e}")
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldTestSetGenerator:
    """Generates gold test set using GPT for evaluation"""
    
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_test_questions(self, num_questions: int = 10) -> List[Dict[str, str]]:
        """Generate test questions and reference answers about Large Language Models"""
        
        prompt = f"""Generate {num_questions} diverse questions about Large Language Models (LLMs) that would be suitable for a RAG system evaluation. 
        For each question, provide a comprehensive reference answer based on current research.

        The questions should cover different aspects of LLMs such as:
        - Training methodologies
        - Architecture and design
        - Applications and use cases
        - Challenges and limitations
        - Performance evaluation
        - Ethical considerations
        - Recent developments

        Format the response as a JSON array where each object has:
        - "question": The question text
        - "reference_answer": A comprehensive answer (2-3 paragraphs)
        - "context": A brief description of what context would be relevant

        Example format:
        [
            {{
                "question": "What are the main challenges in training large language models?",
                "reference_answer": "Training large language models presents several significant challenges...",
                "context": "Information about computational requirements, data quality, and optimization techniques"
            }}
        ]

        Make sure the questions are specific and the answers are detailed and accurate."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            # Extract JSON from the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            json_str = content[json_start:json_end]
            
            test_data = json.loads(json_str)
            logger.info(f"Generated {len(test_data)} test questions")
            return test_data
            
        except Exception as e:
            logger.error(f"Error generating test questions: {e}")
            # Fallback to predefined questions
            return self._get_fallback_questions()
    
    def _get_fallback_questions(self) -> List[Dict[str, str]]:
        """Fallback test questions if GPT generation fails"""
        return [
            {
                "question": "What are the main challenges in training large language models?",
                "reference_answer": "Training large language models presents several significant challenges including computational requirements, data quality and quantity, optimization difficulties, and ethical considerations. The computational cost of training models with billions of parameters requires massive GPU clusters and can cost millions of dollars. Data quality is crucial as models learn from the patterns in their training data, and biased or low-quality data can lead to biased outputs. Optimization challenges include avoiding overfitting, managing gradient flow in deep networks, and finding the right hyperparameters. Additionally, there are ethical concerns about the environmental impact of training, potential misuse of generated content, and the concentration of AI capabilities in large tech companies.",
                "context": "Information about computational requirements, data quality, optimization techniques, and ethical considerations in LLM training"
            },
            {
                "question": "How do transformer architectures work in large language models?",
                "reference_answer": "Transformer architectures form the backbone of modern large language models through their attention mechanism and parallel processing capabilities. The key innovation is the self-attention mechanism, which allows the model to focus on different parts of the input sequence when processing each token. This enables the model to capture long-range dependencies and understand context better than previous architectures like RNNs. Transformers consist of an encoder-decoder structure (in models like BERT and GPT respectively), with multiple layers of attention and feed-forward networks. The attention mechanism computes relationships between all pairs of tokens in the sequence, creating rich representations that capture semantic and syntactic relationships. This architecture allows for efficient parallel processing during training, making it possible to scale to very large models with billions of parameters.",
                "context": "Technical details about attention mechanisms, encoder-decoder structures, and parallel processing in transformers"
            },
            {
                "question": "What are the main applications of large language models?",
                "reference_answer": "Large language models have found applications across numerous domains including natural language processing, code generation, creative writing, and knowledge assistance. In NLP, they power chatbots, language translation, text summarization, and question-answering systems. Code generation models like GitHub Copilot assist developers by suggesting code completions and generating functions based on natural language descriptions. Creative applications include story writing, poetry generation, and content creation for marketing. In education, LLMs serve as tutoring assistants and help with research and writing. Business applications include customer service automation, document analysis, and decision support systems. However, these applications also raise concerns about job displacement, misinformation generation, and the need for human oversight in critical decision-making processes.",
                "context": "Information about various use cases, business applications, and societal impact of LLMs"
            }
        ]

class RAGASEvaluator:
    """Handles RAGAS evaluation of the RAG system"""
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.gold_generator = GoldTestSetGenerator()
    
    def create_evaluation_dataset(self, test_questions: List[Dict[str, str]]) -> Dataset:
        """Create a dataset for RAGAS evaluation"""
        
        questions = []
        ground_truths = []
        contexts = []
        answers = []
        
        for test_item in test_questions:
            try:
                # Get RAG system response
                rag_response = self.rag_pipeline.query(test_item['question'])
                
                # Extract context documents safely
                context_docs = []
                if 'context_documents' in rag_response and rag_response['context_documents']:
                    for doc in rag_response['context_documents']:
                        if isinstance(doc, dict) and 'text' in doc:
                            context_docs.append(doc['text'])
                
                questions.append(test_item['question'])
                ground_truths.append(test_item['reference_answer'])
                contexts.append(context_docs)
                answers.append(rag_response['answer'])
            except Exception as e:
                logger.error(f"Error processing test question '{test_item['question']}': {e}")
                # Skip this item if there's an error
                continue
        
        # Create dataset only if we have data
        if not questions:
            logger.error("No valid evaluation data could be created")
            return None
            
        try:
            dataset = Dataset.from_dict({
                'question': questions,
                'answer': answers,
                'contexts': contexts,
                'ground_truth': ground_truths
            })
            return dataset
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            return None
    
    def evaluate_rag_system(self, num_test_questions: int = 10) -> Dict[str, float]:
        """Evaluate the RAG system using RAGAS metrics"""
        
        if not RAGAS_AVAILABLE:
            return {'error': 'RAGAS evaluation framework is not available. Please install ragas and datasets packages.'}
        
        if Dataset is None:
            return {'error': 'Datasets package is not available. Please install datasets package.'}
        
        # Check if OpenAI API key is available and valid
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found. RAGAS evaluation requires OpenAI API key.")
            return {
                'error': 'OpenAI API key required for RAGAS evaluation. Please set a valid OPENAI_API_KEY environment variable.',
                'suggestion': 'You can get an API key from https://platform.openai.com/api-keys'
            }
        
        # Check if API key is a project key (sk-proj-) which is not supported
        if api_key.startswith('sk-proj-'):
            logger.warning("Project-based OpenAI API key detected. RAGAS requires a standard API key.")
            return {
                'error': 'Project-based OpenAI API key is not supported for RAGAS evaluation.',
                'suggestion': 'Please use a standard OpenAI API key (starting with sk-) instead of a project-based key (starting with sk-proj-)',
                'note': 'You can get a standard API key from https://platform.openai.com/api-keys'
            }
        
        try:
            logger.info("Generating test questions...")
            test_questions = self.gold_generator.generate_test_questions(num_test_questions)
            
            logger.info("Creating evaluation dataset...")
            dataset = self.create_evaluation_dataset(test_questions)
            
            if dataset is None:
                return {'error': 'Failed to create evaluation dataset. Check logs for details.'}
            
            logger.info("Running RAGAS evaluation...")
            
            # Define metrics
            metrics = [
                faithfulness,      # Measures factual consistency of the generated answer
                answer_relevancy, # Measures how relevant the answer is to the question
                context_precision, # Measures how precise the retrieved context is
                context_recall,   # Measures how well the context covers the answer
                ContextRelevance  # Measures how relevant the context is to the question
            ]
            
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics
            )
            
            # Extract scores - handle different result formats
            scores = {}
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            try:
                if hasattr(result, 'to_pandas'):
                    # New RAGAS format
                    df = result.to_pandas()
                    logger.info(f"DataFrame columns: {df.columns.tolist()}")
                    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'context_relevance']:
                        if metric in df.columns:
                            scores[metric] = float(df[metric].mean())
                elif hasattr(result, '__getitem__') and not hasattr(result, '__getattr__'):
                    # Dictionary-like access
                    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'context_relevance']:
                        if metric in result:
                            scores[metric] = float(result[metric])
                else:
                    # Try to access as attributes or properties
                    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'context_relevance']:
                        try:
                            if hasattr(result, metric):
                                value = getattr(result, metric)
                                if hasattr(value, '__call__'):
                                    # It's a method, call it
                                    value = value()
                                scores[metric] = float(value)
                        except Exception as e:
                            logger.warning(f"Could not extract {metric}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error extracting scores: {e}")
                # Fallback: return empty scores
                scores = {}
            
            # Calculate overall score
            if scores:
                scores['overall'] = sum(scores.values()) / len(scores)
            else:
                scores['overall'] = 0.0
            
            logger.info("Evaluation completed successfully")
            return scores
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {'error': str(e)}
    
    def save_evaluation_results(self, scores: Dict[str, float], filename: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        
        results = {
            'scores': scores,
            'timestamp': str(pd.Timestamp.now()),
            'metrics_explanation': {
                'faithfulness': 'Measures factual consistency of the generated answer',
                'answer_relevancy': 'Measures how relevant the answer is to the question',
                'context_precision': 'Measures how precise the retrieved context is',
                'context_recall': 'Measures how well the context covers the answer',
                'context_relevance': 'Measures how relevant the context is to the question'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filename}")
    
    def print_evaluation_summary(self, scores: Dict[str, float]):
        """Print a formatted summary of evaluation results"""
        
        if 'error' in scores:
            print(f"Evaluation failed: {scores['error']}")
            return
        
        print("\n" + "="*50)
        print("RAG SYSTEM EVALUATION RESULTS")
        print("="*50)
        
        for metric, score in scores.items():
            if metric != 'overall':
                print(f"{metric.replace('_', ' ').title()}: {score:.3f}")
        
        print("-"*50)
        print(f"Overall Score: {scores['overall']:.3f}")
        print("="*50)
        
        # Interpretation
        print("\nInterpretation:")
        if scores['overall'] >= 0.8:
            print("🟢 Excellent performance - RAG system is working very well")
        elif scores['overall'] >= 0.6:
            print("🟡 Good performance - RAG system is working well with room for improvement")
        elif scores['overall'] >= 0.4:
            print("🟠 Fair performance - RAG system needs significant improvements")
        else:
            print("🔴 Poor performance - RAG system requires major improvements")

# Example usage
if __name__ == "__main__":
    from rag_pipeline import RAGPipeline
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(rag)
    
    # Run evaluation
    scores = evaluator.evaluate_rag_system(num_test_questions=5)
    
    # Print results
    evaluator.print_evaluation_summary(scores)
    
    # Save results
    evaluator.save_evaluation_results(scores)
