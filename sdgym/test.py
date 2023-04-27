import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sdv.metadata.single_table import SingleTableMetadata

from sdgym.synthesizers.uniform import UniformSynthesizer

data = pd.DataFrame({
    'a': np.random.normal(size=10000),
    'b': np.random.choice(['a', 'b', 'c'], size=10000, p=[.1, .2, .7])
})
metadata = SingleTableMetadata()
metadata.add_column('a', sdtype='numerical')
metadata.add_column('b', sdtype='categorical')
synthesizer = UniformSynthesizer()
obj = synthesizer.get_trained_synthesizer(data, metadata)
samples = synthesizer.sample_from_synthesizer(obj, 10000)
samples.plot(kind='hist')
plt.show()
print(sum(samples == 'a'))
print(sum(samples == 'b'))
print(sum(samples == 'c'))
