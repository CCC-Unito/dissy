# dissy
A Python Library to compute Disagreement Systematicity (σ)

As described in Basile (2026): [A Measure of Systematic Disagreement](https://aclanthology.org/2026.nlperspectives-1.6/)

## Installation

``
pip install git+https://github.com/CCC-Unito/dissy.git
``
## Example

The sigma() function will try to guess the input type: a valid path to a CSV file, a Pandas Dataframe, or a Numpy array.

### With a CSV file

The file needs three columns named:
- instance
- annotator
- label
  
``
from dissy import sigma
sigma("sample_data.csv")
0.8
``

### With a DataFrame (instance, annotator, label)

The DataFrame needs three columns named:
- instance
- annotator
- label
  
``
from dissy import sigma
data = pd.DataFrame({
    'instance': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'annotator': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'label': [1, 1, 0, 1, 1, 1, 1, 0, 0]
})
sigma(data)
1.0
``

### With a NumPy array

The matrix should have shape *num_annotators* X *num_instances*

``
from dissy import sigma
sample_array = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
print(sigma(sample_array))
1.0
``

