# internship

conda create -n mamba310 python=3.10
conda activate mamba310

pip install torch==2.6.0
<!-- pip install transformers==4.39.3 tokenizers==0.18.0 -->
pip install transformers==4.39.3 tokenizers==0.15.2

pip install scipy
pip install causal-conv1d   # 수정된 setup.py 기반으로 설치 (pip install .) at causal-conv1d lib
pip install mamba           # 수정된 setup.py 기반으로 설치 (pip install .) at mamba lib