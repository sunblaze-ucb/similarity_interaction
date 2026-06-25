import torch
import numpy as np
from typing import List, Dict, Optional
import torch.nn.functional as F
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Please install: pip install transformers")
    sys.exit(1)


class LLMEntropyCalculator:
    """Calculate entropy of LLM vocabulary given input text."""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device: Optional[str] = None):
        """
        Initialize with a specific model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        print(f"Loading tokenizer for: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model: {model_name}")
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  # Automatically distributes model across available GPUs
                torch_dtype=torch.bfloat16
                
            )
            
            # Move to device if not using device_map or 8bit
            # if "device_map" not in model_kwargs and not load_in_8bit:
            #     self.model = self.model.to(self.device)
            
            # Store the actual device the model is on
            # This handles cases where device_map="auto" puts it somewhere else
            if hasattr(self.model, 'device'):
                self.model_device = self.model.device
            else:
                # Get device from first parameter
                self.model_device = next(self.model.parameters()).device
                
            print(f"Model actually loaded on: {self.model_device}")
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure you have access to the model (may need to accept license)")
            print("2. Try: huggingface-cli login")
            print("3. Check GPU memory if using CUDA")
            raise
    
    def calculate_entropy_at_position(self, 
                                     input_text: str, 
                                     position: int = -1,
                                     temperature: float = 1.0) -> Dict:
        """
        Calculate entropy at a specific position in the text.
        
        Args:
            input_text: The context text
            position: Token position (-1 for last position)
            temperature: Temperature for scaling logits (1.0 = no scaling)
            
        Returns:
            Dictionary with entropy and additional info
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        
        # Move inputs to model device
        input_ids = inputs["input_ids"].to(self.model_device)
        attention_mask = inputs["attention_mask"].to(self.model_device) if "attention_mask" in inputs else None
        
        # Get model outputs
        with torch.no_grad():
            model_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            
            outputs = self.model(**model_inputs)
            logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
        
        # Select logits at specified position
        if position == -1:
            position = logits.shape[1] - 1
        
        logits_at_position = logits[0, position, :] / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits_at_position, dim=-1)
        
        # Calculate entropy in nats (numerically stable)
        # Only calculate for non-zero probabilities
        probs = probs[probs > 0]
        entropy_nats = -torch.sum(probs * torch.log(probs)).item()
        
        # Calculate entropy in bits
        entropy_bits = entropy_nats / np.log(2)
        
        # Get top predictions 
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
        
        return {
            "entropy_nats": entropy_nats,
            "entropy_bits": entropy_bits,
            "position": position,
            "top_predictions": list(zip(top_tokens, top_probs.cpu().numpy().tolist()))
        }
    
    def calculate_sequence_entropy(self, 
                                  input_text: str,
                                  return_per_token: bool = False,
                                  temperature: float = 1.0) -> Dict:
        """
        Calculate entropy for each position in a sequence.
        
        Args:
            input_text: The context text
            return_per_token: If True, return entropy for each token
            temperature: Temperature for scaling logits
            
        Returns:
            Dictionary with entropy statistics
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True        )
        
        # Move to model device
        input_ids = inputs["input_ids"].to(self.model_device)
        attention_mask = inputs["attention_mask"].to(self.model_device) if "attention_mask" in inputs else None
        
        # Get tokens for display (keep on CPU)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get model outputs
        with torch.no_grad():
            model_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
                
            outputs = self.model(**model_inputs)
            logits = outputs.logits / temperature
        
        entropies_nats = []
        entropies_bits = []
        
        # Calculate entropy at each position
        for i in range(logits.shape[1]):
            logits_at_position = logits[0, i, :]
            probs = F.softmax(logits_at_position, dim=-1)
            
            # Calculate entropy (numerically stable)
            probs_nonzero = probs[probs > 0]
            entropy_nats = -torch.sum(probs_nonzero * torch.log(probs_nonzero)).item()
            entropy_bits = entropy_nats / np.log(2)
            
            entropies_nats.append(entropy_nats)
            entropies_bits.append(entropy_bits)
        
        results = {
            "mean_entropy_nats": np.mean(entropies_nats),
            "mean_entropy_bits": np.mean(entropies_bits),
            "std_entropy_nats": np.std(entropies_nats),
            "std_entropy_bits": np.std(entropies_bits),
            "max_entropy_nats": np.max(entropies_nats),
            "min_entropy_nats": np.min(entropies_nats),
            "total_tokens": len(tokens),
            "text_length": len(input_text)
        }
        
        if return_per_token:
            results["per_token_entropy"] = [
                {
                    "token": token,
                    "entropy_nats": ent_nats,
                    "entropy_bits": ent_bits
                }
                for token, ent_nats, ent_bits in zip(tokens, entropies_nats, entropies_bits)
            ]
        
        return results
    
    def calculate_conditional_entropy(self,
                                     context: str,
                                     continuation: str,
                                     temperature: float = 1.0) -> Dict:
        """
        Calculate the average entropy of generating continuation given context.
        
        Args:
            context: The conditioning context
            continuation: The text to evaluate
            temperature: Temperature for scaling logits
            
        Returns:
            Dictionary with entropy information
        """
        full_text = context + continuation
        
        # Tokenize separately to identify boundaries
        context_inputs = self.tokenizer(context, return_tensors="pt")
        full_inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        
        # Move to model device
        context_ids = context_inputs["input_ids"].to(self.model_device)
        full_ids = full_inputs["input_ids"].to(self.model_device)
        full_attention_mask = full_inputs["attention_mask"].to(self.model_device) if "attention_mask" in full_inputs else None
        
        context_length = context_ids.shape[1]
        
        # Get model outputs for full text
        with torch.no_grad():
            model_inputs = {"input_ids": full_ids}
            if full_attention_mask is not None:
                model_inputs["attention_mask"] = full_attention_mask
                
            outputs = self.model(**model_inputs)
            logits = outputs.logits / temperature
        
        entropies_nats = []
        entropies_bits = []
        
        # Calculate entropy only for continuation positions
        # Note: position i predicts token i+1
        for i in range(context_length - 1, min(logits.shape[1] - 1, full_ids.shape[1] - 1)):
            logits_at_position = logits[0, i, :]
            probs = F.softmax(logits_at_position, dim=-1)
            
            # Calculate entropy (numerically stable)
            probs_nonzero = probs[probs > 0]
            entropy_nats = -torch.sum(probs_nonzero * torch.log(probs_nonzero)).item()
            entropy_bits = entropy_nats / np.log(2)
            
            entropies_nats.append(entropy_nats)
            entropies_bits.append(entropy_bits)
        
        return {
            "mean_entropy_nats": np.mean(entropies_nats) if entropies_nats else 0.0,
            "mean_entropy_bits": np.mean(entropies_bits) if entropies_bits else 0.0,
            "std_entropy_nats": np.std(entropies_nats) if entropies_nats else 0.0,
            "continuation_tokens": len(entropies_nats),
            "context_length": context_length
        }
    
    def calculate_utterance_cross_entropy(self, text: str) -> Dict:
            """
            Calculate cross-entropy (negative log-likelihood) of a specific utterance.
            This is: H(y) = Σ -log p(y^j | y^0...y^{j-1})
            
            Args:
                text: The utterance text
                
            Returns:
                Dictionary with cross-entropy metrics
            """
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True            )
            
            input_ids = inputs["input_ids"].to(self.model_device)
            attention_mask = inputs["attention_mask"].to(self.model_device) if "attention_mask" in inputs else None
            
            # Get model outputs
            with torch.no_grad():
                model_inputs = {"input_ids": input_ids}
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
                
                outputs = self.model(**model_inputs)
                logits = outputs.logits  # [batch, seq_len, vocab]
            
            # Calculate negative log likelihood
            total_nll = 0.0
            token_nlls = []
            
            # For each token position (except the first)
            for i in range(input_ids.shape[1] - 1):
                # Logits at position i predict token at position i+1
                logits_at_i = logits[0, i, :]
                actual_next_token = input_ids[0, i + 1]
                
                # Get log probability of the actual next token
                log_probs = F.log_softmax(logits_at_i, dim=-1)
                token_nll = -log_probs[actual_next_token].item()
                
                total_nll += token_nll
                token_nlls.append(token_nll)
            
            # Convert to bits
            total_nll_bits = total_nll / np.log(2)
            
            return {
                "total_cross_entropy_nats": total_nll,
                "total_cross_entropy_bits": total_nll_bits,
                "mean_cross_entropy_nats": total_nll / len(token_nlls) if token_nlls else 0,
                "mean_cross_entropy_bits": total_nll_bits / len(token_nlls) if token_nlls else 0,
                "num_tokens": len(token_nlls),
                "per_token_nll": token_nlls  # To inspect individual tokens
            }

    def calculate_mutual_information(self,
                                    text1: str,
                                    text2: str,
                                    temperature: float = 1.0) -> Dict:
        """
        Calculate mutual information between two texts.
        
        I(X;Y) = H(Y) - H(Y|X)
        
        Args:
            text1: First text (X)
            text2: Second text (Y)
            temperature: Temperature for scaling logits
            
        Returns:
            Dictionary with mutual information metrics
        """
        # H(Y) - entropy of text2 alone
        h_y = self.calculate_sequence_entropy(text2, temperature=temperature)["mean_entropy_nats"]
        h_y_bits = h_y / np.log(2)
        # H(Y|X) - conditional entropy of text2 given text1
        h_y_given_x = self.calculate_conditional_entropy(text1, text2, temperature=temperature)["mean_entropy_nats"]
        h_y_given_x_bits = h_y_given_x / np.log(2)
        # Mutual information
        mi_nats = h_y - h_y_given_x
        mi_bits = mi_nats / np.log(2)
        
        # Normalized MI (by H(Y))
        normalized_mi = mi_nats / h_y if h_y > 0 else 0
        
        return {
            "mutual_information_nats": mi_nats,
            "mutual_information_bits": mi_bits,
            "normalized_mi": normalized_mi,
            "h_y": h_y,
            "h_y_bits": h_y_bits,
            "h_y_given_x": h_y_given_x,
            "h_y_given_x_bits": h_y_given_x_bits
        }


def test_basic_functionality():
    """Test the calculator with simple examples."""
    print("LLM Vocabulary Entropy Calculator")
    print("=" * 50)
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    calculator = LLMEntropyCalculator(model_name)
    
    # Cross entropy
    utterance = "What is the capital of France?"
    ce_result = calculator.calculate_utterance_cross_entropy(utterance)
    print(f"Utterance: '{utterance}'")
    print(f"Total cross-entropy: {ce_result['total_cross_entropy_nats']:.4f} nats (natural log) -- Cross-entropy bits: {ce_result['total_cross_entropy_bits']:.4f} bits (log2)")
    print(f"Mean cross-entropy per token: {ce_result['mean_cross_entropy_nats']:.4f} nats (natural log) -- Mean cross-entropy bits per token: {ce_result['mean_cross_entropy_bits']:.4f} bits (log2)")
    print(f"Number of tokens: {ce_result['num_tokens']}")

    # Test 1: Single position entropy

    # text = "The United States of"
    # print(f"Entropy at end of text '{text}'")
    # result = calculator.calculate_entropy_at_position(text, temperature=1)
    # print(f"Text: '{text}'")
    # print(f"Entropy: {result['entropy_nats']} nats ({result['entropy_bits']} bits)")
    # print("Top predictions:")
    # for token, prob in result['top_predictions']:
    #     print(f"  '{token}': {prob:.5f}")
    
    # # Test 2: Sequence entropy

    # print("Sequence entropy")
    # text = "Machine learning is a powerful tool."
    # # text = "hopfdnlf this is higgg entr."
    # results = calculator.calculate_sequence_entropy(text)
    # print(f"Text: '{text}'")
    # print(f"Mean entropy: {results['mean_entropy_nats']:.4f} nats (natural log) ({results['mean_entropy_bits']:.4f} bits (log2))")
    # # print(f"Std deviation: {results['std_entropy_nats']:.4f} nats")
    # print(f"Token count: {results['total_tokens']}")
    

    # # Test 3: Conditional entropy

    # print("\nTest 3: Conditional entropy")
    # context = "The weather today is"
    # continuation = " sunny and warm"
    # result = calculator.calculate_conditional_entropy(context, continuation)
    # print(f"Context: '{context}'")
    # print(f"Continuation: '{continuation}'")
    # print(f"Conditional entropy: {result['mean_entropy_nats']:.4f} nats ({result['mean_entropy_bits']:.4f} bits)")
    

    # Test 4: Mutual information

    # print("Mutual information")
    # text1 = "Paris is the capital of France."
    # # text2 = "Washington is the capital of the United States."
    # text2 = "grrrr bow123"
    # result = calculator.calculate_mutual_information(text1, text2)
    # print(f"Text 1: '{text1}'")
    # print(f"Text 2: '{text2}'")
    # print(f"Mutual information: {result['mutual_information_nats']:.4f} nats (natural log)")
    # print(f"Mutual information: {result['mutual_information_bits']:.4f} bits (log2)")
    # print(f"H(Y): {result['h_y']:.4f} nats (natural log) -- {result['h_y_bits']:.4f} bits (log2) ")
    # print(f"H(Y|X): {result['h_y_given_x']:.4f} nats (natural log) -- {result['h_y_given_x_bits']:.4f} bits (log2)")
    # print(f"Normalized MI: {result['normalized_mi']:.4f}")'



if __name__ == "__main__":
    test_basic_functionality()