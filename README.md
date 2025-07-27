# 🥥 CoCoNut: Continual & Competition Network with Updatable Templates

> **Where Competition meets Continual Learning for Intelligent Adaptation**

## 🎯 Project Overview

CoCoNut is an advanced **continual learning system** for touchless palmprint recognition, implementing a novel **2-stage learning strategy** that combines robust pre-training with intelligent online adaptation.

### 🔥 Key Innovation: Controlled Batch Composition Learning

| Stage | Method | Goal |
|-------|--------|------|
| **Stage 1 (Pretrain)** | Hybrid Loss (ArcFace + SupCon) | Build robust feature space |
| **Stage 2 (Adapt)** | SupCon with controlled batch composition | Smart adaptation to new data |

---

## 🏗️ Architecture

```
CoCoNut = CCNet + Intelligent Replay Buffer + Controlled Batch Composition
```

### Core Components

- **🧠 CCNet**: Gabor-based competition network for palmprint feature extraction
- **🎯 Controlled Batch Composition**: Precise positive/hard sample ratios for optimal learning
- **🔄 Intelligent Replay**: Diversity-based experience replay with Faiss indexing
- **🔪 Headless Support**: Optional classification head removal for open-set recognition

---

## 📊 Mathematical Foundation

### Stage 1 - Hybrid Loss
```
L_pretrain = α × L_ArcFace + β × L_SupCon
where α = 0.8, β = 0.2 (empirically validated)
```

### Stage 2 - Controlled SupCon Loss
```
L_adapt = -Σ log(p_i) with controlled batch composition
- Target positive ratio: 30% (configurable)
- Hard mining ratio: 30% (configurable)
- Smart replay buffer sampling
```

**Batch Composition Parameters**:
- `continual_batch_size = 10` (separate from pretrain batch size)
- `target_positive_ratio = 0.3` (30% positive pairs)
- `hard_mining_ratio = 0.3` (30% hard samples)

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# GPU environment (recommended)
pip install torch torchvision faiss-gpu
pip install -r requirements.txt

# CPU environment  
pip install torch torchvision faiss-cpu
pip install -r requirements.txt
```

### 2️⃣ Data Preparation

```bash
./data/train_tongji.txt     # Pre-training data
./data/test_tongji.txt      # Validation data  
/path/to/live/palm_data     # Online adaptation data
```

### 3️⃣ Execution

#### Stage 1: Pre-training
```bash
python pretrain.py
```
- Uses `config/pretrain_config.yaml`
- Applies hybrid loss (ArcFace + SupCon)
- Builds robust feature space foundation

#### Stage 2: Online Adaptation
```bash
python run_coconut.py
```
- Uses `config/adapt_config.yaml`
- Applies controlled batch composition learning
- Performs continual adaptation with precise ratios

---

## 📁 Project Structure

```
CoCoNut/
├── 📁 config/
│   ├── pretrain_config.yaml    # Stage 1 configuration
│   ├── adapt_config.yaml       # Stage 2 configuration  
│   └── config_parser.py        # Unified config parser
├── 📁 models/
│   ├── ccnet_model.py          # Competition Network with Headless support
│   ├── trainer.py              # Pre-training trainer
│   └── config.py               # Model configurations
├── 📁 framework/
│   ├── coconut.py              # Main continual learning system
│   ├── losses.py               # SupCon loss functions
│   ├── replay_buffer.py        # Intelligent experience replay
│   └── config.py               # Framework configurations
├── 📁 datasets/
│   ├── palm_dataset.py         # Palmprint dataset loader
│   └── config.py               # Dataset configurations
├── 📁 evaluation/
│   └── eval_utils.py           # Performance evaluation tools
├── 🐍 pretrain.py              # Stage 1 execution script
├── 🐍 run_coconut.py           # Stage 2 execution script
└── 📄 requirements.txt         # Dependencies
```

---

## 🎛️ Configuration

### Design Philosophy Documentation

All configuration files include comprehensive design rationales:

```yaml
Design_Documentation:
  stage1_philosophy: "Stable hybrid learning for robust generalization"
  stage2_philosophy: "Controlled batch composition for optimal continual learning"  
  loss_strategy: "ArcFace+SupCon → Controlled SupCon transition"
  batch_strategy: "Precise positive/hard ratios for learning efficiency"
```

---

## 📈 Performance Monitoring

### Real-time Batch Composition Analysis

| Feature | Description |
|---------|-------------|
| **🎯 Controlled Ratios** | Precise positive/hard sample composition |
| **📊 Batch Tracking** | Real-time monitoring of batch statistics |
| **🔬 Diversity Analysis** | Buffer diversity and sampling efficiency |
| **📈 Learning Progress** | Continuous adaptation effectiveness |

### Example Output

```
[Controlled] 📊 Batch Composition Analysis:
   Target batch size: 10
   Positive pairs: 1 pairs (2 samples, 20.0%)
   Hard samples: 3 samples (30.0%)
   Regular samples: 5 samples (50.0%)
   Achievement: 0.7x target positive ratio

[Buffer] 📊 Diversity Statistics:
   Total samples: 45/50
   Unique users: 12
   Diversity score: 0.87
```

---

## 🔬 Scientific Foundation

### ✅ Mathematical Accuracy
- **SupCon Loss**: Standard supervised contrastive learning
- **Controlled Composition**: Novel batch sampling strategy
- **Real-time Verification**: Continuous batch composition validation

### ✅ Experimental Reproducibility  
- **Deterministic Seeding**: Fixed random seeds for reproducible results
- **Configuration Traceability**: Complete documentation of all parameters
- **Statistical Analysis**: Comprehensive performance tracking

---

## 📊 Performance Results

| Metric | Improvement |
|--------|------------|
| **Accuracy** | ↗️ Consistent adaptation performance |
| **EER** | ↘️ Reduced error rates |
| **Learning Efficiency** | 🚀 Controlled batch composition optimization |
| **Memory Usage** | 💾 Efficient buffer management |

---

## 🔪 Headless Mode Features

### Open-Set Recognition Support

```yaml
# Headless Configuration
headless_mode: true
verification_method: "metric"  # cosine similarity
similarity_threshold: 0.5
compression_dim: 128  # 2048 → 128 compression
```

**Benefits**:
- **Memory Efficiency**: 16x compression (2048→128)
- **Open-Set Ready**: No fixed class constraints
- **Metric Verification**: Cosine similarity-based matching
- **Flexible Deployment**: Adaptable to new users

---

## 📚 References

1. **SupCon Foundation**: Supervised Contrastive Learning for robust feature learning
2. **CCNet Architecture**: Competition Network for palmprint recognition
3. **Continual Learning**: Experience replay with catastrophic forgetting prevention
4. **Faiss Integration**: Efficient similarity search and diversity maintenance

---

## 🤝 Contributing

We welcome contributions! Please ensure:

- ✅ Batch composition implementations maintain precise ratios
- ✅ All new features include comprehensive documentation  
- ✅ Configuration changes maintain backward compatibility
- ✅ Test coverage for critical functionality

### Development Setup

```bash
git clone https://github.com/eunsu0325/CL-CCNet
cd coconut
pip install -r requirements.txt
pip install -e .  # Development install
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SupCon Methodology**: Supervised Contrastive Learning community
- **CCNet Architecture**: Competition Network innovations  
- **Faiss Library**: Efficient similarity search capabilities
- **Open-source Community**: Deep learning framework contributors

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/eunsu0325/coconut/issues)
- **Discussions**: [GitHub Discussions](https://github.com/eunsu/coconut/discussions)
- **Email**: kuma1577@naver.com

---

Made with ❤️ by the CoCoNut Team