# ScrewsIK Toolbox

This repository contains a complete Python implementation of inverse kinematics algorithms based on **screw theory**, originally written in MATLAB and translated to Python by Hernan Trullo.

The implementation is based on the following reference:

> **"Screw Theory in Robotics: An Illustrated and Practicable Introduction to Modern Mechanics"**  
> José M. Pardos-Gotor – CRC Press, 2022

---

## 🧠 Overview

This toolbox provides numerical solutions for inverse kinematics problems using:

- **Paden-Kahan subproblems (PK1, PK2, PK3)**
- **Pardos-Gotor subproblems (PG1 through PG8)**
- **Worked examples from Chapter 4** of the reference book, including applications for industrial robots such as the ABB IRB120 and SCARA IRB 910SC.

This library is designed for research, educational, and experimental purposes.

---

## 🛠️ Project Structure

toolbox_screws_theory/
├── toolbox/
│ ├── paden_kahan.py # PK subproblems
│ ├── pardos_gotor.py # PG subproblems
│ └── utils.py # Common utilities
├── pruebas_pk_pg/ # Test scripts
└── README.md


---

## 📦 Installation

```bash
git clone https://github.com/HernanTrullo/toolbox_screws_theory.git
cd toolbox_screws_theory
pip install -r requirements.txt
