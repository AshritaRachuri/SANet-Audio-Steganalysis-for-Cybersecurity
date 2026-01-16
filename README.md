# 🔐 SANet — Audio Steganalysis for Cybersecurity

SANet is an AI-driven audio steganalysis system designed to detect hidden or manipulated information within speech audio.  
The project focuses on **cybersecurity and digital forensics**, addressing covert data leakage through voice channels that often bypass traditional security tools.

Unlike conventional malware detection that focuses on files or images, SANet targets **speech-based covert communication**, making it robust against compression, noise, and real-world audio distortions.

---

## 🧠 Problem Statement

Cyber attackers increasingly exploit **audio channels** (VoIP calls, voice messages, recordings) to secretly transmit information.  
Most existing security systems fail to analyze or detect these hidden patterns in speech signals.

SANet aims to:
- Detect hidden information embedded in speech audio  
- Remain robust under compression and environmental noise  
- Support real-world cybersecurity and forensic use cases  

---
## 🏗 System Architecture

The following diagram illustrates the high-level architecture of SANet, showing the interaction between
audio input, preprocessing, deep learning inference, and result generation.

![SANet Architecture](./assets/images/architecture.png)


## 🔄 Workflow

The workflow below outlines the step-by-step execution pipeline of the SANet system,
from audio upload to steganalysis detection and result visualization.

![SANet Workflow](./assets/images/workflow.png)


## 🚀 Key Applications

- **Cybersecurity** – Prevent covert data exfiltration through audio channels  
- **Telecommunications** – Secure VoIP and call monitoring systems  
- **Digital Forensics & Law Enforcement** – Analyze suspicious or tampered audio evidence  

---

## 🛠 Tech Stack

| Component | Technology |
|---------|------------|
| Backend | Python, Flask |
| Deep Learning | PyTorch |
| Audio Processing | Librosa |
| Frontend | React |
| Deployment | Docker, AWS (optional) |

---

## 🧩 System Architecture

1. User uploads speech audio  
2. Audio preprocessing & feature extraction  
3. Deep learning model analyzes hidden patterns  
4. Classification result (clean / suspicious)  
5. Outputs & visualizations generated  

---

## 🔄 Workflow

1. Audio input is uploaded via frontend or API  
2. Backend preprocesses audio (spectrograms / features)  
3. SANet model performs steganalysis  
4. Detection results are returned to the user  

---

## ✨ Features

- Audio-based covert content detection  
- Robust to compression and noise  
- Modular backend architecture  
- Scalable for real-world deployment  
- Suitable for cybersecurity and forensic pipelines  

---



