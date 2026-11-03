# Asistente de clasificación de correos / tickets

Proyecto que clasifica mensajes de soporte en tres categorías —`urgente`,
`consulta`, `queja`— y **compara dos enfoques** sobre el mismo conjunto de 100
mensajes etiquetados a mano:

1. **Baseline: árbol de decisión sobre palabras clave.** En dos variantes:
   - un árbol de reglas `if/else` legible, y
   - un `DecisionTreeClassifier` de scikit-learn entrenado sobre variables de
     conteo de palabras clave.
2. **LLM** (NVIDIA NIM vía API compatible con OpenAI), al que se le pide
   clasificar cada mensaje mediante un *prompt*.

Ambos se miden con **precision, recall y F1** (por clase y promedio *macro*),
que son las métricas propias de un problema de clasificación.

## Conocimiento reutilizado de los notebooks del curso

- `ML_exercise.ipynb` → `DecisionTreeClassifier`, `train_test_split` y el uso de
  precision/recall/F1 en clasificación.
- `05_ejercicio_nvidia_documentos.ipynb` → cliente `OpenAI` sobre NVIDIA NIM, API
  key leída de `.env` con `python-dotenv`, y `chat.completions.create`.

> Este proyecto es nuevo: **no modifica** ninguno de los notebooks existentes.

## Archivos

- `07_clasificador_tickets.ipynb` — notebook principal (end-to-end).
- `tickets_dataset.csv` — 100 mensajes etiquetados a mano (`mensaje`, `etiqueta`).

## Cómo ejecutar

```bash
pip install -q pandas scikit-learn openai python-dotenv matplotlib
```

Abre `07_clasificador_tickets.ipynb` y ejecuta las celdas en orden.

### Usar el LLM real (opcional)

Crea un archivo `.env` junto al notebook con tu clave de NVIDIA Build:

```text
NVIDIA_API_KEY="tu_api_key"
```

Sin API key el notebook funciona igual: el clasificador LLM cae automáticamente
al árbol de reglas, de modo que la comparación se ejecuta de principio a fin.
