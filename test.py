import pandas as pd

df = pd.DataFrame(index=list(range(10)), columns=list(range(10)))

df[0,0] = 0
print()