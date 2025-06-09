"""
Simple test script for the Educational LLM
"""

import torch
import tiktoken
from educational_llm import EducationalLLM, LLMTrainer, TextDataset, create_sample_config
from torch.utils.data import DataLoader

def test_model_basic():
    """Test basic model functionality"""
    print("Testing Educational LLM...")
    
    # Set device
    device = torch.device('cpu')  # Use CPU for testing
    print(f"Using device: {device}")
    
    # Create smaller config for testing
    config = {
        'vocab_size': 50257,      # Use GPT-2 vocab size to match tokenizer
        'd_model': 128,           # Smaller model dimension
        'n_layers': 2,            # Fewer layers
        'n_heads': 4,             # Fewer heads
        'max_seq_length': 256,    # Shorter sequences
        'd_ff': 512,              # Smaller feed-forward
        'dropout': 0.1
    }
    
    print(f"Model configuration: {config}")
    
    # Initialize tokenizer (we'll use simple token indices for testing)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create model
    model = EducationalLLM(config)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))  # Use smaller range for testing
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Output logits shape: {logits.shape}")
        
        # Test with attention weights
        logits, attention_weights = model(input_ids, return_attention_weights=True)
        print(f"Number of attention weight tensors: {len(attention_weights)}")
        print(f"Attention weights shape: {attention_weights[0].shape}")
    
    print("✓ Basic model test passed!")
    
    # Test text generation (simple)
    trainer = LLMTrainer(model, tokenizer, device)
    
    # Use a very simple generation test
    print("\nTesting text generation...")
    try:
        # Start with a simple token sequence
        input_text = "Hello"
        generated = trainer.generate_text(input_text, max_length=10, temperature=1.0)
        print(f"Generated text: {generated}")
        print("✓ Text generation test passed!")
    except Exception as e:
        print(f"Text generation test failed: {e}")
    
    print("\nAll tests completed!")

def test_with_real_data():
    """Test with a larger sample of real text data"""
    print("\n" + "="*50)
    print("TESTING WITH REAL TEXT DATA")
    print("="*50)
    
    # Larger sample text
    sample_text = """
    Once upon a time, in a land far away, there lived a young programmer who dreamed of creating intelligent machines.
    Every day, she would study the mysteries of artificial intelligence and machine learning.
    She learned about neural networks, transformers, and the magic of deep learning.
    The world of AI was vast and complex, filled with mathematical equations and computational challenges.
    But she persevered, knowing that each line of code brought her closer to her dream.
    Language models fascinated her the most - how they could understand and generate human language.
    She studied the attention mechanism, the transformer architecture, and the principles of natural language processing.
    With each passing day, her knowledge grew, and her models became more sophisticated.
    The future of artificial intelligence looked bright, and she was determined to be part of it.
    Through hard work and dedication, she would contribute to the advancement of technology.
    """ * 5  # Repeat to have more data
    
    # Create config
    config = create_sample_config()
    config['d_model'] = 256  # Smaller for faster testing
    config['n_layers'] = 3
    config['n_heads'] = 4
    
    # Set device
    device = torch.device('cpu')
    
    # Initialize tokenizer and model
    tokenizer = tiktoken.get_encoding("gpt2")
    model = EducationalLLM(config)
    trainer = LLMTrainer(model, tokenizer, device)
    
    print(f"Model has {model.count_parameters():,} parameters")
    
    # Create dataset
    dataset = TextDataset(sample_text, tokenizer, max_length=64, stride=32)
    print(f"Dataset created with {len(dataset)} samples")
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Test one training step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        print("\nTesting training step...")
        model.train()
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            loss.backward()
            optimizer.step()
            
            print(f"Training loss: {loss.item():.4f}")
            break  # Just test one step
        
        print("✓ Training step test passed!")
        
        # Test generation
        print("\nTesting generation with trained model...")
        generated_text = trainer.generate_text(
            "Once upon a time", 
            max_length=20, 
            temperature=0.8
        )
        print(f"Generated: {generated_text}")
        print("✓ Generation test passed!")
    
    else:
        print("Dataset is empty - text might be too short")

if __name__ == "__main__":
    test_model_basic()
    test_with_real_data()