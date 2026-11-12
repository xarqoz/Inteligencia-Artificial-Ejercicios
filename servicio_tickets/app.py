"""API web del clasificador de tickets (FastAPI) + frontend interactivo.

Expone el clasificador híbrido como un servicio HTTP y sirve una interfaz web
sencilla para probarlo desde el navegador.

Ejecutar:
    uvicorn app:app --reload

Luego abre en el navegador:
    http://127.0.0.1:8000/          (interfaz interactiva)
    http://127.0.0.1:8000/docs      (documentación OpenAPI)

Probar por consola:
    curl -X POST http://127.0.0.1:8000/clasificar \
         -H "Content-Type: application/json" \
         -d '{"mensaje": "El servidor está caído, no podemos facturar"}'
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from clasificador import ClasificadorTickets, CATEGORIAS, evidencia_keywords, normaliza

# Se crea una sola vez al arrancar: detecta el modelo LLM disponible (si hay key).
clasificador = ClasificadorTickets()

app = FastAPI(
    title="Clasificador de tickets de soporte",
    description="Clasifica mensajes en urgente / consulta / queja / otro y sugiere enrutamiento.",
    version="2.0.0",
)

# CORS abierto para que la interfaz funcione también si se abre el HTML aparte.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


class TicketEntrada(BaseModel):
    mensaje: str = Field(..., min_length=1, description="Texto del ticket o correo.")


class TicketSalida(BaseModel):
    mensaje: str
    etiqueta: str
    fuente: str
    confianza_reglas: int
    requiere_revision: bool
    prioridad: str
    equipo: str
    accion: str
    evidencia: dict


@app.get("/", include_in_schema=False)
def index():
    """Sirve la interfaz web interactiva."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/salud")
def salud():
    """Estado del servicio y si el LLM está activo."""
    return {
        "estado": "ok",
        "categorias": list(CATEGORIAS),
        "llm_disponible": clasificador.llm_disponible,
        "modelo": getattr(clasificador.llm, "modelo", None),
        "umbral_confianza": clasificador.umbral,
    }


@app.post("/clasificar", response_model=TicketSalida)
def clasificar(ticket: TicketEntrada):
    """Clasifica un ticket y devuelve la acción de enrutamiento sugerida."""
    resultado = clasificador.clasifica(ticket.mensaje).to_dict()
    # Adjuntamos la evidencia de palabras clave por categoría, útil para la UI.
    resultado["evidencia"] = evidencia_keywords(normaliza(ticket.mensaje))
    return resultado
