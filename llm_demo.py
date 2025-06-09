#!/usr/bin/env python3
"""
Educational LLM Demo Script

This script demonstrates all the key features of the Educational LLM:
1. Model architecture overview
2. Training on sample data
3. Text generation with different strategies
4. Model analysis and visualization
5. Saving and loading models

Run with: python llm_demo.py
"""

import torch
import tiktoken
import json
from educational_llm import (
    EducationalLLM, LLMTrainer, TextDataset, 
    create_sample_config
)
from torch.utils.data import DataLoader

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def analyze_model_architecture(model):
    """Analyze and display model architecture details"""
    print_section("MODEL ARCHITECTURE ANALYSIS")
    
    print(f"📊 Model Statistics:")
    print(f"   • Total parameters: {model.count_parameters():,}")
    print(f"   • Model dimension: {model.d_model}")
    print(f"   • Number of layers: {model.n_layers}")
    print(f"   • Number of attention heads: {model.n_heads}")
    print(f"   • Vocabulary size: {model.vocab_size}")
    print(f"   • Maximum sequence length: {model.max_seq_length}")
    
    # Calculate parameter distribution
    total_params = model.count_parameters()
    embedding_params = model.token_embedding.weight.numel() + model.position_embedding.weight.numel()
    output_params = model.lm_head.weight.numel()
    transformer_params = total_params - embedding_params - output_params
    
    print(f"\n📈 Parameter Distribution:")
    print(f"   • Embeddings: {embedding_params:,} ({100*embedding_params/total_params:.1f}%)")
    print(f"   • Transformer blocks: {transformer_params:,} ({100*transformer_params/total_params:.1f}%)")
    print(f"   • Output head: {output_params:,} ({100*output_params/total_params:.1f}%)")
    
    # Memory estimation (rough)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32 parameter
    print(f"\n💾 Estimated memory usage: {memory_mb:.1f} MB")

def demonstrate_attention_patterns(model, tokenizer, text):
    """Demonstrate attention pattern visualization"""
    print_section("ATTENTION PATTERN ANALYSIS")
    
    # Encode text
    tokens = tokenizer.encode(text)[:20]  # Limit to 20 tokens
    input_ids = torch.tensor(tokens).unsqueeze(0)
    
    # Get attention weights
    with torch.no_grad():
        logits, attention_weights = model(input_ids, return_attention_weights=True)
    
    # Decode tokens for display
    token_strings = [tokenizer.decode([token]) for token in tokens]
    
    print(f"📝 Analyzing text: '{text}'")
    print(f"🔤 Tokens ({len(tokens)}): {token_strings[:10]}{'...' if len(tokens) > 10 else ''}")
    
    # Show attention patterns for the first layer, first head
    if attention_weights:
        attn = attention_weights[0][0, 0]  # First batch, first head
        print(f"\n🎯 Attention patterns (Layer 1, Head 1):")
        print("   (Each row shows what the token attends to)")
        
        # Show top 3 attention weights for first few tokens
        for i in range(min(5, len(tokens))):
            token = token_strings[i]
            top_indices = torch.topk(attn[i], k=3)[1]
            top_tokens = [token_strings[idx] for idx in top_indices if idx < len(token_strings)]
            print(f"   • '{token}' → {top_tokens}")

def training_demonstration(model, tokenizer, sample_text):
    """Demonstrate the training process"""
    print_section("TRAINING DEMONSTRATION")
    
    # Create trainer
    device = torch.device('cpu')
    trainer = LLMTrainer(model, tokenizer, device)
    
    # Create dataset
    dataset = TextDataset(sample_text, tokenizer, max_length=64, stride=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"📚 Dataset: {len(dataset)} training samples")
    print(f"📊 Batch size: 2")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    print(f"\n🏃 Training for 3 steps...")
    
    # Train for a few steps
    model.train()
    losses = []
    
    for step, (input_ids, targets) in enumerate(dataloader):
        if step >= 3:  # Only train for 3 steps for demo
            break
            
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"   Step {step + 1}: Loss = {loss.item():.4f}")
    
    print(f"✅ Training complete! Loss improved from {losses[0]:.4f} to {losses[-1]:.4f}")
    
    return trainer

def generation_showcase(trainer, prompts):
    """Showcase different text generation strategies"""
    print_section("TEXT GENERATION SHOWCASE")
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎲 Prompt {i+1}: '{prompt}'")
        
        # Greedy generation (temperature = 0.1)
        greedy_text = trainer.generate_text(
            prompt, max_length=15, temperature=0.1, top_k=1
        )
        print(f"   🎯 Greedy: {greedy_text[len(prompt):]}")
        
        # Creative generation (higher temperature)
        creative_text = trainer.generate_text(
            prompt, max_length=15, temperature=1.2, top_k=50, top_p=0.9
        )
        print(f"   🎨 Creative: {creative_text[len(prompt):]}")

def model_comparison_demo():
    """Compare different model sizes"""
    print_section("MODEL SIZE COMPARISON")
    
    configs = [
        {"name": "Tiny", "d_model": 64, "n_layers": 2, "n_heads": 2},
        {"name": "Small", "d_model": 128, "n_layers": 4, "n_heads": 4},
        {"name": "Medium", "d_model": 256, "n_layers": 6, "n_heads": 8},
    ]
    
    print("📏 Comparing different model sizes:\n")
    
    for config_info in configs:
        # Create config
        config = create_sample_config()
        config.update({
            'd_model': config_info['d_model'],
            'n_layers': config_info['n_layers'],
            'n_heads': config_info['n_heads'],
            'd_ff': 4 * config_info['d_model']
        })
        
        # Create model
        model = EducationalLLM(config)
        params = model.count_parameters()
        memory_mb = (params * 4) / (1024 * 1024)
        
        print(f"   {config_info['name']:>6}: {params:>10,} params | {memory_mb:>6.1f} MB")

def save_load_demo(trainer):
    """Demonstrate model saving and loading"""
    print_section("MODEL PERSISTENCE DEMO")
    
    model_path = "demo_model.pth"
    
    # Save model
    print(f"💾 Saving model to '{model_path}'...")
    trainer.save_model(model_path)
    
    # Create new trainer and load model
    print(f"📂 Loading model from '{model_path}'...")
    config = trainer.model.config
    new_model = EducationalLLM(config)
    new_trainer = LLMTrainer(new_model, trainer.tokenizer, trainer.device)
    new_trainer.load_model(model_path)
    
    # Test that loaded model works
    test_text = new_trainer.generate_text("Hello", max_length=5)
    print(f"✅ Loaded model test: '{test_text}'")

def main():
    """Main demonstration function"""
    print("🎓 Educational Large Language Model (LLM) Demo")
    print("   A comprehensive PyTorch implementation for learning")
    
    # Setup
    device = torch.device('cpu')
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create model with educational-sized config
    config = create_sample_config()
    config.update({
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 8,
        'd_ff': 1024,
        'max_seq_length': 512
    })
    
    model = EducationalLLM(config)
    
    # Sample text for training
    sample_text = """
    Artificial intelligence is transforming our world in remarkable ways.
    Language models like transformers can understand and generate human text.
    These models learn patterns from vast amounts of text data.
    The attention mechanism allows them to focus on relevant parts of the input.
    Deep learning has revolutionized natural language processing.
    Neural networks with millions of parameters can capture complex relationships.
    Training these models requires careful optimization and large datasets.
    The future of AI holds exciting possibilities for human-computer interaction.
    Machine learning continues to advance at an unprecedented pace.
    Understanding these technologies is crucial for the next generation.
    """ * 3  # Repeat for more training data
    
    # Run demonstrations
    analyze_model_architecture(model)
    
    demonstrate_attention_patterns(
        model, tokenizer, 
        "The quick brown fox jumps over the lazy dog"
    )
    
    trainer = training_demonstration(model, tokenizer, sample_text)
    
    generation_showcase(trainer, [
        "The future of AI",
        "Machine learning",
        "Deep neural networks"
    ])
    
    model_comparison_demo()
    
    save_load_demo(trainer)
    
    print_section("DEMO COMPLETE!")
    print("🎉 You've successfully explored the Educational LLM!")
    print("💡 Key concepts covered:")
    print("   • Transformer architecture")
    print("   • Multi-head attention")
    print("   • Training process")
    print("   • Text generation strategies")
    print("   • Model analysis")
    print("\n📚 Next steps:")
    print("   • Experiment with different model sizes")
    print("   • Try training on your own text data")
    print("   • Explore different generation parameters")
    print("   • Study the attention patterns")

if __name__ == "__main__":
    main()