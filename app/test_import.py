import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from llm_outputs.model_invoke import model_invoke_generate_quiz
    print("Import Successful")
except ImportError as e:
    print(f"Import Failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
