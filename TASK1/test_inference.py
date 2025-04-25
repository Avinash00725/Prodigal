import pytest
from hello_inference import run_inference

def test_run_inference(capsys):
    # Run the inference function and capture output
    run_inference()
    captured = capsys.readouterr()
    # Check that the output contains the expected message
    assert "Hello, Inference!" in captured.out
    # Check that it runs on either cuda or cpu
    assert "Running on cuda" in captured.out or "Running on cpu" in captured.out
