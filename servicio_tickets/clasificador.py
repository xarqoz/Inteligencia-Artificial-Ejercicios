"""Clasificador híbrido de tickets de soporte: urgente / consulta / queja.

Estrategia (la conclusión del notebook 07_clasificador_tickets.ipynb, aplicada):

    1. Un árbol de decisión sobre PALABRAS CLAVE resuelve los casos claros de
       forma instantánea y gratuita.
    2. Solo los casos dudosos (poca evidencia de keywords) se envían al LLM
       (NVIDIA NIM), que entiende el contexto pero cuesta tiempo y tokens.

Así se equilibran costo, latencia y precisión. Si no hay API key configurada,
el servicio funciona igual usando únicamente el árbol de reglas.

Cada clasificación devuelve además una ACCIÓN de enrutamiento sugerida, que es
lo que da valor en un caso real (a quién se asigna el ticket y con qué prioridad).
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass, asdict

# --------------------------------------------------------------------------- #
# 1. Normalización y diccionario de palabras clave
# --------------------------------------------------------------------------- #

CATEGORIAS = ("urgente", "consulta", "queja")


def normaliza(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar palabras clave de forma robusta."""
    texto = str(texto).lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


# Mismo diccionario, ya afinado, que el notebook (baseline ~0.97 de F1 macro).
KEYWORDS = {
    "urgente": [
        "urgente", "urge", "inmediat", "caid", "fuera de servicio", "no funciona",
        "no responde", "no puedo acceder", "critic", "emergencia", "inaccesible",
        "colaps", "bloquead", "se bloqueo", "fuga de datos", "ataque", "incidente",
        "falla", "fallar", "error 500", "perdemos", "perder datos", "riesgo",
        "alerta", "no accedemos", "no podemos", "se cayo", "se perdio",
        "perdida de conexion", "interrupcion", "corte", "dejo de funcionar",
        "no procesa", "no valida", "no abre", "no permite", "no actualiza",
        "saldos incorrectos", "sobrecalentamiento", "comprometid", "actuen ya",
        "necesitamos ayuda ya", "se cierra sola", "no pueden",
    ],
    "queja": [
        "inaceptable", "pesim", "decepcion", "decepcionad", "lamentable", "harto",
        "frustrante", "estafad", "enganad", "enganar", "enganaron", "malisim",
        "terrible", "nadie me", "nadie resuelve", "no resuelven", "nunca",
        "retraso", "roto", "rompio", "no cumple", "reembolso", "compensacion",
        "quiero mi dinero", "inutil", "decepcionante", "molesto", "grosero",
        "abandonan", "no corresponde", "incompleto", "sin solucion",
        "sin arreglarlo", "sin avisar", "me cobraron", "siguen cobrando",
        "me siento", "me prometieron", "me enviaron", "me cambiaron", "me prometio",
        "mal servicio", "mala imagen", "mala", "no soluciona", "inexistente",
        "no reconozco", "no se presento", "tirado", "sucio", "mas barato",
        "perdiendo la confianza", "pesima experiencia",
    ],
    "consulta": [
        "como", "donde", "cuando", "cuanto", "cual", "cuales", "podrian", "podria",
        "quisiera saber", "me gustaria saber", "me interesa saber", "es posible",
        "puedo", "tienen", "ofrecen", "quisiera", "me gustaria", "disponible",
        "consultar", "diferencia entre", "requisitos", "horario", "queria",
        "puedo obtener", "donde puedo", "cual es", "cuanto tiempo", "que incluye",
        "que metodos",
    ],
}

PRIORIDAD = ["urgente", "queja", "consulta"]


def cuenta_keywords(texto_norm: str) -> dict[str, int]:
    """Cuántas palabras clave de cada categoría aparecen en el texto."""
    return {cat: sum(1 for kw in kws if kw in texto_norm) for cat, kws in KEYWORDS.items()}


def arbol_reglas(texto: str) -> str:
    """Árbol de decisión sobre palabras clave (baseline rápido y gratuito)."""
    tn = normaliza(texto)
    c = cuenta_keywords(tn)
    es_pregunta = tn.strip().endswith("?")

    if c["urgente"] >= 1 and c["urgente"] >= c["queja"]:
        return "urgente"
    if c["queja"] >= 1 and c["queja"] >= c["consulta"]:
        return "queja"
    if es_pregunta or c["consulta"] >= 1:
        return "consulta"
    mejor = max(PRIORIDAD, key=lambda k: c[k])
    return mejor if c[mejor] > 0 else "consulta"


# --------------------------------------------------------------------------- #
# 2. Clasificador LLM (NVIDIA NIM) — opcional
# --------------------------------------------------------------------------- #

MODELOS_CANDIDATOS = [
    "nvidia/nemotron-3.5-bolt-e4b",  # verificado: funciona en la cuenta
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "meta/llama-3.1-8b-instruct",
]

SYSTEM_PROMPT = """Eres un asistente que clasifica mensajes de soporte al cliente.
Debes asignar a cada mensaje EXACTAMENTE una de estas tres categorías:

- urgente: incidentes que requieren acción inmediata; algo está caído, no
  funciona, hay riesgo de seguridad o pérdida, o bloquea la operación.
- consulta: una pregunta o solicitud de información; el cliente quiere saber algo
  y no reporta un fallo ni expresa enfado.
- queja: el cliente expresa insatisfacción o reclama por una mala experiencia,
  aunque no sea un incidente técnico urgente.

Responde ÚNICAMENTE con una palabra en minúsculas: urgente, consulta o queja.
No agregues explicaciones ni signos de puntuación."""


def _parsea_etiqueta(texto: str) -> str:
    """Normaliza la salida del modelo a una de las categorías válidas."""
    t = normaliza(texto or "")
    for cat in CATEGORIAS:
        if cat in t:
            return cat
    return "consulta"


class ClasificadorLLM:
    """Envuelve el cliente de NVIDIA NIM y detecta un modelo válido al iniciar."""

    def __init__(self) -> None:
        self.cliente = None
        self.modelo = None
        self.disponible = False
        self._inicializa()

    def _inicializa(self) -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            return
        try:
            from openai import OpenAI
        except Exception:
            return

        cliente = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        for nombre in MODELOS_CANDIDATOS:
            try:
                cliente.chat.completions.create(
                    model=nombre,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                self.cliente = cliente
                self.modelo = nombre
                self.disponible = True
                return
            except Exception:
                continue

    def clasifica(self, mensaje: str) -> str:
        if not self.disponible:
            raise RuntimeError("LLM no disponible")
        resp = self.cliente.chat.completions.create(
            model=self.modelo,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mensaje},
            ],
            temperature=0,
            max_tokens=4,
        )
        return _parsea_etiqueta(resp.choices[0].message.content)


# --------------------------------------------------------------------------- #
# 3. Enrutamiento: qué hacer con cada categoría (el valor de negocio)
# --------------------------------------------------------------------------- #

ACCIONES = {
    "urgente": {
        "prioridad": "alta",
        "equipo": "guardia / on-call",
        "accion": "crear alerta inmediata y asignar al equipo de guardia",
    },
    "consulta": {
        "prioridad": "baja",
        "equipo": "soporte nivel 1",
        "accion": "responder con FAQ o asignar a soporte de nivel 1",
    },
    "queja": {
        "prioridad": "media",
        "equipo": "retención / supervisor",
        "accion": "escalar a un supervisor del equipo de retención",
    },
}


@dataclass
class ResultadoClasificacion:
    mensaje: str
    etiqueta: str
    fuente: str          # "reglas" o "llm"
    confianza_reglas: int
    prioridad: str
    equipo: str
    accion: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 4. Clasificador híbrido (lo que se usa en producción)
# --------------------------------------------------------------------------- #

class ClasificadorTickets:
    """Combina el árbol de reglas (casos claros) con el LLM (casos dudosos)."""

    def __init__(self, umbral_confianza: int = 2, usar_llm: bool = True) -> None:
        self.umbral = umbral_confianza
        self.llm = ClasificadorLLM() if usar_llm else None

    @property
    def llm_disponible(self) -> bool:
        return bool(self.llm and self.llm.disponible)

    def clasifica(self, mensaje: str) -> ResultadoClasificacion:
        tn = normaliza(mensaje)
        conteos = cuenta_keywords(tn)
        confianza = max(conteos.values())
        etiqueta_reglas = arbol_reglas(mensaje)

        # Caso claro -> reglas. Caso dudoso -> LLM (si está disponible).
        if confianza >= self.umbral or not self.llm_disponible:
            etiqueta, fuente = etiqueta_reglas, "reglas"
        else:
            try:
                etiqueta, fuente = self.llm.clasifica(mensaje), "llm"
            except Exception:
                etiqueta, fuente = etiqueta_reglas, "reglas"

        info = ACCIONES[etiqueta]
        return ResultadoClasificacion(
            mensaje=mensaje,
            etiqueta=etiqueta,
            fuente=fuente,
            confianza_reglas=confianza,
            prioridad=info["prioridad"],
            equipo=info["equipo"],
            accion=info["accion"],
        )
