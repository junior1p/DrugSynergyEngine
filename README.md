# DrugSynergyEngine

Pure Python drug combination synergy analysis pipeline.

## Features
- Hill equation dose-response fitting (scipy.optimize)
- Bliss independence model
- Loewe additivity model
- HSA (Highest Single Agent) model
- Synergy landscape visualization
- 20 clinically relevant drug pairs

## Usage
```bash
pip install numpy scipy pandas matplotlib
python drug_synergy_engine.py
```

## Results (10 drugs, 20 pairs, 8x8 matrices)
- 4 synergistic pairs (Bliss>8)
- Top: Olaparib+Venetoclax (Bliss=9.5, PARP+BCL2)
- Hill equation mean R2=0.997
