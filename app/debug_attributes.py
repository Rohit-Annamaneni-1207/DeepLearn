import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import llm_outputs.model_invoke as model_invoke

print("Attributes in model_invoke:")
print(dir(model_invoke))

if hasattr(model_invoke, 'model_invoke_generate_quiz'):
    print("SUCCESS: model_invoke_generate_quiz found.")
else:
    print("FAILURE: model_invoke_generate_quiz NOT found.")
