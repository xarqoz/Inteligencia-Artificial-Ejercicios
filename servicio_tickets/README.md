# Servicio de clasificación de tickets (ejemplo de caso real)

Este directorio lleva el clasificador del notebook
`07_clasificador_tickets.ipynb` **de prototipo a servicio real**: un API web con
**interfaz interactiva** que recibe el texto de un ticket y devuelve su categoría
(`urgente` / `consulta` / `queja` / `otro`) junto con la **acción de enrutamiento**
sugerida (prioridad, equipo y qué hacer).

## ⚠️ Aviso de seguridad (léelo antes de compartir)

- **Nunca compartas tu archivo `.env` real.** Contiene tu API key de NVIDIA
  (empieza por `nvapi-`). El `.gitignore` evita que `.env` se suba a Git, pero
  **NO protege un archivo ZIP**: si comprimes la carpeta, el `.env` va dentro.
- Antes de enviar un ZIP (al profesor, a compañeros, a cualquier parte):
  **borra el `.env`** del ZIP y, si ya lo compartiste, **revoca esa clave y genera
  una nueva** en https://build.nvidia.com/settings/api-keys.
- Para configurar la clave usa siempre una **copia** de `.env.example`.

## Enfoque híbrido

Aplica la conclusión del notebook: **no es LLM _o_ reglas, es ambos**.

```
Ticket ──► Árbol de palabras clave (gratis, instantáneo)
              │
              ├─ caso claro (evidencia suficiente) ───► usar esa etiqueta
              │
              ├─ caso dudoso (poca evidencia) ────────► preguntar al LLM (NVIDIA NIM)
              │
              └─ sigue sin estar claro ───────────────► marcar "revisar" (humano en el bucle)
```

- El árbol de keywords resuelve la mayoría de casos al instante y sin costo.
- El LLM se reserva para los mensajes ambiguos.
- **No hay fallback silencioso.** Cuando ni reglas ni LLM dan una respuesta
  clara, el ticket se marca `requiere_revision = true` en lugar de asignarle una
  categoría (y una prioridad baja) a ciegas. Esto protege contra el peor error:
  un ticket **urgente** clasificado por descarte como consulta.
- Si no hay API key, el servicio funciona igual con solo el árbol de reglas.

Cada etiqueta dispara una **acción**:

| Categoría  | Prioridad | Equipo               | Acción                                      |
|------------|-----------|----------------------|---------------------------------------------|
| urgente    | alta      | guardia / on-call    | crear alerta inmediata y asignar a guardia  |
| consulta   | baja      | soporte nivel 1      | responder con FAQ / asignar a nivel 1       |
| queja      | media     | retención/supervisor | escalar a un supervisor de retención        |
| otro       | baja      | sin acción           | saludo/agradecimiento/spam: archivar        |
| *revisar*  | media     | triaje humano        | clasificación incierta: la revisa una persona |

## Interfaz interactiva

Al levantar el servidor y abrir `http://127.0.0.1:8000/` tienes una página web
donde puedes:

- escribir un ticket o elegir un ejemplo con un clic;
- ver la categoría con color e icono, la prioridad, el equipo y la acción;
- ver si la decisión vino del **árbol de reglas**, del **LLM** o si **requiere
  revisión** humana;
- ver un gráfico de barras con la **evidencia de palabras clave** por categoría;
- ver en la cabecera si el **LLM está activo** y qué modelo se usa.

## Archivos

- `clasificador.py` — lógica central (normalización, árbol de reglas, cliente
  LLM, clasificador híbrido y enrutamiento).
- `app.py` — API FastAPI (`/`, `/clasificar`, `/salud`) que además sirve la UI.
- `static/index.html` — interfaz web interactiva (sin dependencias externas).
- `demo.py` — prueba por consola, sin levantar servidor.
- `requirements.txt`, `.env.example`, `.gitignore`.

## Cómo ejecutar

```bash
cd servicio_tickets
pip install -r requirements.txt
```

### Opción A — Interfaz web (recomendado)

```bash
uvicorn app:app --reload
```

Abre en el navegador: **http://127.0.0.1:8000/**
Documentación OpenAPI: http://127.0.0.1:8000/docs

### Opción B — Demo por consola

```bash
python demo.py
```

### Opción C — Llamar al API directamente

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
  "requiere_revision": false,
  "prioridad": "alta",
  "equipo": "guardia / on-call",
  "accion": "crear alerta inmediata y asignar al equipo de guardia",
  "evidencia": {"urgente": 2, "queja": 0, "consulta": 0, "otro": 0}
}
```

### Activar el LLM (opcional)

```bash
cp .env.example .env
# edita .env y pega tu NVIDIA_API_KEY (NO subas este archivo ni lo metas en un ZIP)
```

El servicio detecta automáticamente un modelo válido de NVIDIA NIM al arrancar.

## Limitaciones conocidas (honestas)

- **La "confianza" es solo un conteo de palabras clave, no está calibrada.** El
  umbral por defecto (2) es una decisión razonable pero no demostrada; en
  producción se ajustaría con un conjunto de validación (igual que `max_depth`
  en `ML_exercise.ipynb`).
- **El ~0.97 de F1 del notebook es optimista.** Se midió sobre los mismos datos
  con los que se afinaron las palabras clave, así que probablemente sobreestima
  el rendimiento en tickets reales (riesgo de sobreajuste).
- **Prompt injection:** el texto del ticket lo escribe un tercero. El system
  prompt se reforzó para tratar el mensaje como contenido y no como
  instrucciones, y la salida se restringe a una palabra; aun así conviene
  monitorear.
- **Arranque:** al iniciar, el servicio prueba los modelos candidatos con una
  llamada de red cada uno hasta hallar uno válido (se puede desactivar con
  `verificar_al_iniciar=False`).

## De este ejemplo a producción

Este servicio es un **ejemplo didáctico**. Para un despliegue real faltaría:

- **Monitoreo continuo**: medir precision/recall/F1 sobre tickets reales
  periódicamente (no solo una vez) y reentrenar/ajustar keywords cuando bajen.
- **Humano en el bucle** para los casos críticos (ya implementado como estado
  `revisar`, pero habría que conectarlo a una cola de triaje real).
- **Privacidad**: los tickets reales contienen datos personales; cuidar qué se
  envía a una API externa.
- **Escala**: caché, límites de tasa, colas y autenticación del API.
