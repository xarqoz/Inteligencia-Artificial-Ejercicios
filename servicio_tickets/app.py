"""API web del clasificador de tickets (FastAPI).

Expone el clasificador híbrido como un servicio HTTP, de modo que un sistema de
mesa de ayuda (Zendesk, Freshdesk, Jira, un formulario web...) pueda enviar el
texto de un ticket y recibir su categoría + la acción de enrutamiento sugerida.

Ejecutar:
    uvicorn app:app --reload

Probar:
    curl -X POST http://127.0.0.1:8000/clasificar \
         -H "Content-Type: application/json" \
         -d '{"mensaje": "El servidor está caído, no podemos facturar"}'

Documentación interactiva: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from clasificador import ClasificadorTickets, CATEGORIAS

# Se crea una sola vez al arrancar: detecta el modelo LLM disponible (si hay key).
clasificador = ClasificadorTickets()

app = FastAPI(
    title="Clasificador de tickets de soporte",
    description="Clasifica mensajes en urgente / consulta / queja y sugiere enrutamiento.",
    version="1.0.0",
)


class TicketEntrada(BaseModel):
    mensaje: str = Field(..., min_length=1, description="Texto del ticket o correo.")


class TicketSalida(BaseModel):
    mensaje: str
    etiqueta: str
    fuente: str
    confianza_reglas: int
    prioridad: str
    equipo: str
    accion: str


@app.get("/salud")
def salud():
    """Estado del servicio y si el LLM está activo."""
    return {
        "estado": "ok",
        "categorias": list(CATEGORIAS),
        "llm_disponible": clasificador.llm_disponible,
        "modelo": getattr(clasificador.llm, "modelo", None),
    }


@app.post("/clasificar", response_model=TicketSalida)
def clasificar(ticket: TicketEntrada):
    """Clasifica un ticket y devuelve la acción de enrutamiento sugerida."""
    return clasificador.clasifica(ticket.mensaje).to_dict()
