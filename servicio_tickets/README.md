# Servicio de clasificación de tickets (ejemplo de caso real)

Este directorio lleva el clasificador del notebook
`07_clasificador_tickets.ipynb` **de prototipo a servicio real**: un API web que
recibe el texto de un ticket y devuelve su categoría (`urgente` / `consulta` /
`queja`) junto con la **acción de enrutamiento** sugerida (prioridad, equipo y
qué hacer).

## Enfoque híbrido

Aplica la conclusión del notebook: **no es LLM _o_ reglas, es ambos**.

```
Ticket ──► Árbol de palabras clave (gratis, instantáneo)
              │
              ├─ caso claro (confianza alta) ─────────► usar esa etiqueta
              │
              └─ caso dudoso (poca evidencia) ────────► preguntar al LLM (NVIDIA NIM)
```

- El árbol de keywords resuelve la mayoría de casos al instante y sin costo.
- El LLM se reserva para los mensajes ambiguos, equilibrando costo, latencia y
  precisión.
- Si no hay API key, el servicio funciona igual con solo el árbol de reglas.

Cada etiqueta dispara una **acción**:

| Categoría | Prioridad | Equipo              | Acción                                      |
|-----------|-----------|---------------------|---------------------------------------------|
| urgente   | alta      | guardia / on-call   | crear alerta inmediata y asignar a guardia  |
| consulta  | baja      | soporte nivel 1     | responder con FAQ / asignar a nivel 1       |
| queja     | media     | retención/supervisor| escalar a un supervisor de retención        |

## Archivos

- `clasificador.py` — lógica central (normalización, árbol de reglas, cliente
  LLM, clasificador híbrido y enrutamiento).
- `app.py` — API web con FastAPI (`/clasificar`, `/salud`).
- `demo.py` — prueba por consola, sin levantar servidor.
- `requirements.txt`, `.env.example`, `.gitignore`.

## Cómo ejecutar

```bash
cd servicio_tickets
pip install -r requirements.txt
```

### Opción A — Demo por consola (rápido)

```bash
python demo.py
```

### Opción B — API web

```bash
uvicorn app:app --reload
```

Luego, en otra terminal:

```bash
curl -X POST http://127.0.0.1:8000/clasificar \
     -H "Content-Type: application/json" \
     -d '{"mensaje": "El servidor está caído, no podemos facturar"}'
```

Respuesta de ejemplo:

```json
{
  "mensaje": "El servidor está caído, no podemos facturar",
  "etiqueta": "urgente",
  "fuente": "reglas",
  "confianza_reglas": 2,
  "prioridad": "alta",
  "equipo": "guardia / on-call",
  "accion": "crear alerta inmediata y asignar al equipo de guardia"
}
```

Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

### Activar el LLM (opcional)

```bash
cp .env.example .env
# edita .env y pega tu NVIDIA_API_KEY
```

El servicio detecta automáticamente un modelo válido de NVIDIA NIM al arrancar.

## De este ejemplo a producción

Este servicio es un **ejemplo didáctico**. Para un despliegue real faltaría:

- **Monitoreo continuo**: medir precision/recall/F1 sobre tickets reales
  periódicamente (no solo una vez), y reentrenar/ajustar keywords cuando bajen.
- **Humano en el bucle** para los casos `urgente`, donde un error cuesta caro.
- **Privacidad**: los tickets reales contienen datos personales; cuidar qué se
  envía a una API externa.
- **Escala**: caché, límites de tasa, colas y autenticación del API.
