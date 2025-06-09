"""
Educational Large Language Model (LLM) Implementation in PyTorch

This implementation provides a comprehensive, well-documented transformer-based LLM
for educational purposes. It includes:
- Complete transformer architecture with detailed comments
- Training loop with loss tracking
- Text generation with different sampling strategies
- Model saving/loading functionality
- Performance monitoring
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
import json
import os
from typing import Optional, List, Dict, Any
import time


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need"
    """
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism with causal masking for autoregressive generation
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask for autoregressive generation
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer block with multi-head attention and feed-forward network
    Includes residual connections and layer normalization
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and layer norm
        attn_output, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x, attn_weights


class EducationalLLM(nn.Module):
    """
    Complete transformer-based Language Model for educational purposes
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.max_seq_length = config['max_seq_length']
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional embeddings (learnable)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=config.get('d_ff', 4 * self.d_model),
                dropout=config.get('dropout', 0.1)
            ) for _ in range(self.n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def create_causal_mask(self, seq_len: int, device: torch.device):
        """Create causal mask for autoregressive attention"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, input_ids: torch.Tensor, return_attention_weights: bool = False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            if return_attention_weights:
                attention_weights.append(attn_weights)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if return_attention_weights:
            return logits, attention_weights
        return logits
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextDataset(Dataset):
    """
    Dataset class for text data with sliding window tokenization
    """
    def __init__(self, text: str, tokenizer, max_length: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        
        # Create overlapping sequences
        self.sequences = []
        for i in range(0, len(tokens) - max_length + 1, stride):
            self.sequences.append(tokens[i:i + max_length])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        # Input is sequence[:-1], target is sequence[1:]
        return sequence[:-1], sequence[1:]


class LLMTrainer:
    """
    Trainer class for the Educational LLM
    """
    def __init__(self, model: EducationalLLM, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                   scheduler=None) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for input_ids, targets in dataloader:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 1.0, top_k: int = 50, 
                     top_p: float = 0.9) -> str:
        """
        Generate text using different sampling strategies
        """
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions for the last position
                logits = self.model(generated)[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit the end-of-text token (if it exists in vocab)
                try:
                    eos_token_id = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
                    if next_token.item() == eos_token_id:
                        break
                except:
                    pass  # No EOS token, continue generating
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text
    
    def save_model(self, filepath: str):
        """Save model and training state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Model loaded from {filepath}")


def create_sample_config():
    """Create a sample configuration for the model"""
    return {
        'vocab_size': 50257,      # GPT-2 tokenizer vocabulary size
        'd_model': 512,           # Model dimension
        'n_layers': 6,            # Number of transformer layers
        'n_heads': 8,             # Number of attention heads
        'max_seq_length': 1024,   # Maximum sequence length
        'd_ff': 2048,             # Feed-forward dimension
        'dropout': 0.1            # Dropout rate
    }


def main():
    """
    Main function demonstrating the Educational LLM usage
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model configuration
    config = create_sample_config()
    print(f"Model configuration: {config}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create model
    model = EducationalLLM(config)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Create trainer
    trainer = LLMTrainer(model, tokenizer, device)
    
    # Sample text for demonstration
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for training our educational language model.
    Artificial intelligence and machine learning are transforming the world in unprecedented ways.
    Large language models like GPT have shown remarkable capabilities in understanding and generating human language.
    """
    
    # Create dataset and dataloader
    dataset = TextDataset(sample_text, tokenizer, max_length=128, stride=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Generate text before training
    print("\n" + "="*50)
    print("GENERATION BEFORE TRAINING:")
    print("="*50)
    sample_generation = trainer.generate_text("The future of AI", max_length=50)
    print(sample_generation)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Train for a few epochs (demo)
    print("\n" + "="*50)
    print("TRAINING:")
    print("="*50)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        train_loss = trainer.train_epoch(dataloader, optimizer)
        print(f"Average training loss: {train_loss:.4f}")
    
    # Generate text after training
    print("\n" + "="*50)
    print("GENERATION AFTER TRAINING:")
    print("="*50)
    sample_generation = trainer.generate_text("The future of AI", max_length=50)
    print(sample_generation)
    
    # Save model
    trainer.save_model("educational_llm.pth")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()