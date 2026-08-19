# DisSy 
A Python Library to compute Disagreement Systematicity (σ, *pron. 'sigma'*)

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

## Theoretical background

In Social Psychology, like/dislike relationships between humans are modeled through signed undirected graphs, where the nodes represent the individuals and and edge between A and B represent their relationship (if present) as positive (+) or negative (-).

The theory of [Structural Balance](https://en.wikipedia.org/wiki/Balance_theory) ([Cartwright and Harary, 1956](https://pubmed.ncbi.nlm.nih.gov/13359597/)) posits that any three entities in relationship with each other (a *triangle*) in such a graph tend towards a balance achieved by either all edges being + or a situation where one edge is + and the other two are -. The other two possible configurations are instead regarded as imbalanced:

![alt text](https://raw.githubusercontent.com/CCC-Unito/dissy/refs/heads/main/SBT.png?raw=true)

[Davis (1967)](https://www.sciencedirect.com/science/chapter/edited-volume/abs/pii/B9780124424500500092?via%3Dihub) applies the notion of structural balance to graphs, calling a balanced graph an undirected signed graph where all triangles (i.e., cycles of length 3) are balanced, and proving that a balanced graph has a unique clustering. I extend this definition to a degree of balancedness, that is, the rate of triangles in an undirected signed graph that are balanced:

$$
\sigma = \frac{(\\\# \text{balanced triangles}) }{(\\\# \text{triangles})}
$$

The outcome of an annotation task is represented as a signed undirected graph, where each node represents an annotator, and the $+ / -$ sign indicates whether the pair agrees ($+$) or disagrees ($-$). The pairwise [Krippendorff's $\alpha$](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) is computed ($\alpha_{i , j}$ where $i$ and $j$ are two annotators) and compared to the mean agreement $\overline\alpha$. The edge ${i, j}$ is given the positive sign (+) if $\alpha_{i , j} \geq \overline\alpha$, or the negative sign (-) otherwise.