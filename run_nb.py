import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('analysis_updated.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')

try:
    ep.preprocess(nb, {'metadata': {'path': './'}})
except Exception as e:
    print(f"Error executing notebook: {e}")

with open('analysis.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Finished execution.")
