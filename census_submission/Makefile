.PHONY: train api test slices clean
PY=python
UVICORN=uvicorn
train: ; $(PY) train_model.py
api:   ; $(UVICORN) main:app --reload
test:  ; pytest -q
slices:; head -n 20 slice_output.txt || echo 'Run `make train` first.'
clean: ; rm -rf model/*.pkl model/metrics.json || true
