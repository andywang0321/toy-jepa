echo "Experiments running on GPUs $CUDA_VISIBLE_DEVICES"

uv sync
uv run scripts/m1_pixel.py
uv run scripts/m2_jepa.py
uv run scripts/m3_stopgrad.py
uv run scripts/m4_ema.py
uv run scripts/m5_sigreg.py

echo "Experiments completed"

