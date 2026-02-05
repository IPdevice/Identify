# Helix: Accurate Internet-Scale Device Identification

<img width="1271" height="255" alt="image" src="https://github.com/user-attachments/assets/8c9cf8d1-7a0d-4750-9193-3ac2436808ac" />

**Helix** is an intelligent device identification system that continuously evolves with the network ecosystem. It leverages Large Language Models (LLMs) through workflow orchestration and multi-agent collaboration to accurately identify and classify network devices, while heuristically maintaining and expanding its rule database to improve coverage and efficiency over time.


## 🎯 Overview

Helix addresses the challenge of device identification in dynamic network environments by combining:

- **Multi-dimensional Information Collection**: Scans IP addresses and collects comprehensive device information
- **LLM-based Intelligent Labeling**: Employs specialized agents for hierarchical device identifcation
- **Heuristic Rule Management**: Automatically maintains and optimizes a rule database for efficient matching
- **Continuous Evolution**: Adapts to new device types and network patterns over time

## 🏗️ System Architecture

Helix operates through a multi-stage pipeline:

1. **Data Collection** : Scans IP addresses and collects multidimensional device information
2. **Rule Matching**: If data matches existing rules, outputs device type according to match strategy
3. **LLM-based Labeling** : Triggers intelligent labeling process for unmatched devices
4. **Semantic Extraction** : Extracts key fields from multidimensional information, with optional Deep Search module for comprehensive content understanding
5. **Hierarchical Identifcation** :
   - **Macro-Class Identification Agent**: Assigns device to one of five major categories
   - **Micro-Class Agents**: Identify subcategories and extend them when needed
6. **Rule Induction** : Derives new rules from identification results, integrating key features with LLM reasoning traces
7. **Rule Database Update**: Applies heuristic mechanisms to ensure rule quality and maintain database integrity

## ✨ Key Features

- **Multi-Agent Collaboration**: Specialized agents for different identifcation tasks
- **Hierarchical Identifcation**: Two-level identifcation (macro-class → micro-class)
- **Automatic Rule Generation**: Extracts and validates rules from LLM reasoning
- **Rule Quality Assurance**: Sample validation, LLM evaluation, and inverted index checks
- **Continuous Learning**: Rule database evolves with new device types
- **High Performance**: C++-accelerated inverted index for fast rule matching

## 📋 Requirements

- Python 3.8 or higher
- LLM API access (OpenAI-compatible)
- Required Python packages (see installation section)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Helix.git
cd Helix
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Set your LLM API keys as environment variables:

```bash
export API_KEY_OPENAI="your-openai-api-key"
export BASE_URL_OPENAI="https://api.openai.com/v1"  # Optional, for custom endpoints
```

Alternatively, you can modify `config.py` directly (not recommended for production).

### 4. Build the Inverted Index

Before running device identification, build the inverted index for rule matching:

```bash
cd step
python build_index_fast.py ../test_data.json --batch_size 5000 --sample_size 200000
```

This creates an optimized index for fast rule matching. The index will be stored in `step/improved_dynamic_index/`.

## 📖 Usage

### Basic Usage

Run device identification on your input data:

```bash
python main.py ./test_data.json
```

### Input Format

The input file should be a JSON file containing device information. Each entry should include multidimensional information such as:
- IP address
- Port information
- Service details
- HTTP headers
- Protocol information
- Geographic information
- ASN information
- Other device-specific fields

Example structure:
```json
[
  {
    "ip": "192.168.1.1",
    "port": 80,
    "service": "http",
    "headers": {...},
    "title": "...",
    ...
  },
  ...
]
```

### Output

The system generates:
- `res_result.json`: Detailed identification results for each device
- `final_label.txt`: Device labels in the format `main_category|subcategory`
- `rules.json`: Updated rule database (automatically maintained)


## 📁 Project Structure

```
Helix/
├── main.py                 # Main entry point and orchestration
├── config.py               # API key configuration
├── rules.json              # Rule database (auto-updated)
├── final_label.txt         # Device labels output
├── step/                   # Core modules
│   ├── build_index_fast.py # Index building utility
│   ├── clean_rules_1.py    # Rule matching and validation
│   ├── clean_rules_2.py    # Rule update and optimization
│   ├── extract_rules.py    # Rule extraction from LLM output
│   └── fast_index_wrapper.py # C++ accelerated index wrapper
├── subclass/               # Subcategory definitions
│   ├── ICSs.txt
│   ├── IoT_devices.txt
│   ├── network_infrastructure.txt
│   ├── Service Devices.txt
│   └── user_terminals.txt
├── ddgs-main/              # Deep search crawler implementation
├── cache/                  # LLM response cache
└── deleted_rules/          # Archive of deleted rules
```

## 🔧 Advanced Configuration

### Customizing Identifcation Categories

Edit the files in `subclass/` to customize or extend device categories:
- `ICSs.txt`: Industrial Control Systems
- `IoT_devices.txt`: IoT devices
- `network_infrastructure.txt`: Network infrastructure devices
- `Service Devices.txt`: Service devices
- `user_terminals.txt`: User terminals


## 📊 Performance

Helix is optimized for performance:
- **C++ Accelerated Indexing**: Fast inverted index for rule matching
- **Batch Processing**: Efficient batch processing of devices
- **Caching**: LLM response caching to reduce API calls

---

**Note**: This is a research prototype. For production use, please ensure proper security measures and API key management.

