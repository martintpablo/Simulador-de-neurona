import streamlit as st

## Objeto

import numpy as np
import math

class Neuron:

  def __init__(self, weights, bias, func):
    self.weights = weights
    self.bias = bias
    self.func = func

  def changeBias(self, n):
    self.bias = n

  def run(self, input_data):
    if (len(input_data) != len(self.weights)):
      return "No tienen el mismo tamaño"
    else:
      y = np.dot(input_data, self.weights) + self.bias

      if self.func == "ReLU":
        if (y <= 0):
          return 0
        else:
          return y

      elif self.func == "Sigmoide":
        y = 1 / (1 + (math.exp(-y)))
        return y

      elif self.func == "Tangente hiperbólica":
        return math.tanh(y)

#######

st.set_page_config(layout="wide")

st.image('https://img.freepik.com/vector-gratis/diagrama-celula-vastago-fondo-blanco_1308-15286.jpg', width=400)

st.title('**Simulador de neurona**')

n_neuronas = st.slider('Elige el número de entradas/pesos que tendrá la neurona', 1, 10, key="n_neuronas")

pesos = []
entradas = []

st.markdown("## Pesos")

cols_w = st.columns(n_neuronas)
for i in range(0, n_neuronas):
    with cols_w[i]:
        pesos.append(st.number_input(f'w{i}', step=0.1, key=f'w{i}'))

st.text(f"w = {pesos}")

st.markdown("## Entradas")

cols_x = st.columns(n_neuronas)
for i in range(0, n_neuronas):
    with cols_x[i]:
        entradas.append(st.number_input(f'x{i}', step=0.1, key=f'x{i}'))

st.text(f"x = {entradas}")

cols_final = st.columns(2)
with cols_final[0]:
    st.markdown("## Sesgo")
    b = st.number_input("Introduce el valor del sesgo", step=0.1, key='sesgo')

with cols_final[1]:
    st.markdown("## Función de activación")
    func = st.selectbox(
    'Elige la función de activación',
    ('Sigmoide', 'ReLU', 'Tangente hiperbólica'))

if st.button('Calcular la salida'):
    neurona = Neuron(weights=pesos, bias=b, func=func)
    x = entradas
    output = neurona.run(input_data=x)
    st.text(f"La salida de la neurona es {output}")
