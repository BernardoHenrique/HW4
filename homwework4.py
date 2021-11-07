# Grupo 117 Aprendizagem HomeWork 4
# Bernardo Castico ist196845
# Hugo Rita ist196870

import pandas as pd
from scipy.io import arff

def main():
    data, dataFinal = [],[]
    data = arff.loadarff('breast.w.arff')[0]
    for i in data:
        dataFinal += [list(i)]
    for i in dataFinal:
        i[9] = i[9[2:]]
    print(dataFinal)
main()