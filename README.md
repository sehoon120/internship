# Internship – Mamba Model Analysis & FPGA SSM Block Implementation

## 📌 Overview

This repository contains my internship work exploring the **Mamba** sequence model architecture and implementing its **SSM (Structured State Space Model) Block** in Verilog for FPGA acceleration.

The project includes:

* Python-based analysis of the **Mamba** model internals
* Extraction of SSM-related parameters and dataflow
* Implementation and verification of **SSM Block** in Verilog
* Simulation-based performance & accuracy comparison with PyTorch FP16

---

## 🛠️ Python Environment Setup

```bash
# Create and activate conda environment
conda create -n mamba-2 python=3.10
conda activate mamba-2

# Install dependencies
pip install torch==2.6.0
pip install transformers==4.39.3 tokenizers==0.15.2
pip install scipy

# # Install causal-conv1d (custom setup.py)
# cd causal-conv1d
# pip install .

# # Install mamba (custom setup.py)
# cd mamba
# pip install .
```

---

## 📂 Repository Structure

```
.
├── Mamba/Mamba-2/mamba2-minimal/   # Clone from mamba2-minimal GitHub
│   ├── verilog/ver4/               # Verilog implementation of SSM Block (Current version)
│   │   ├── pipelined_hC.v
│   │   ├── accumulator.v
│   │   ├── packing.v
│   │   ├── tb_packing_server.v
│   │   └── ...
│   ├── verification_tile_wise.py   # Python-Verilog co-simulation scripts
│   └── verilog/intermediate_datas/ # In/Out datas of SSM Block (FP16 .hex files)
└── README.md
```

---

## 🔍 Key Work

### 1. Python Mamba Analysis

* Studied `mamba2-minimal` implementation
* Traced **EWM (Element-Wise Multiply)**, **EWA (Element-Wise Add)**, and accumulation flow inside the SSM Block
* Extracted parameters: `dt`, `dA`, `B`, `C`, `D`, `x` for hardware input

### 2. Verilog SSM Block Implementation

* Designed modular structure:

    * **`pipelined_hC.v`**  
      Computes  
      `hC = ((d * x) * B_mat + dA * h_prev) * C`  
      Pipelined for higher throughput.

    * **`xD.v`**  
      Calculates  
      `xD = x * D`

    * **`accumulator.v`**  
      Performs summation along the N dimension:  
      `y_tmp = Σ(hC)`

    * **`y_out.v`**  
      Produces the final output:  
      `y_out = y_tmp + xD`

    * **`top.v`**
    Wires all submodules to form the complete **SSM Block**.

    * **`packing.v`**
    Accepts the full-size input, splits it into tiles, and sends each tile to the SSM Block.
    *(Planned change: Move this role to the testbench so `top.v` can have smaller input width.)*

    * **`tb_packing_server.v`**
    Testbench that:

    1. Sends the entire input dataset to `packing.v`
    2. Collects the output from the SSM Block
    3. Saves the output as `.hex` files

* Used **FP16 IEEE-754 half-precision** via Xilinx Floating Point IPs

### 3. Verification

* Co-simulation with Python (torch.float16)
* Achieved:

  * **Max abs error**: 0.007812 (FP16 minimum unit)
  * **Mean abs error**: 0.00022
* Simulation latency reduced by **\~62%** with pipeline improvements

---

## 📈 Performance Summary

| Platform                | Avg SSM Block Runtime |
| ----------------------- | --------------------- |
| CPU (i7-1165G7)         | \~477 µs              |
| GPU (RTX 4070 Ti SUPER) | \~202 µs              |
| Verilog Simulation      | \~149 µs              |

---

## 🚧 Current Issues & Next Steps

* **Issue**: Large in/out port width → synthesis failure
* **Planned Fix**: Store weights in external memory (DDR/HBM) and fetch via DMA with ping-pong buffering
* **Next Steps**:

  1. Decide FPGA board, memory type, bus protocol, host CPU
  2. Implement memory interface for tile-wise data transfer
  3. Generate bitstream and perform hardware in-loop testing

---

## 📎 References

* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
* [mamba2-minimal GitHub](https://github.com/tommyip/mamba2-minimal)

---