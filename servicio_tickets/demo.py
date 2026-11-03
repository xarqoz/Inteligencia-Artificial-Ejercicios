"""Demo sin servidor: clasifica una lista de tickets de ejemplo y muestra
la categoría, la fuente (reglas o LLM) y la acción de enrutamiento sugerida.

Ejecutar:
    python demo.py

Funciona sin API key (usa solo el árbol de reglas). Si configuras
NVIDIA_API_KEY en un archivo .env, los casos dudosos se resolverán con el LLM.
"""

from clasificador import ClasificadorTickets

EJEMPLOS = [
    "El sistema de pagos está caído y no podemos facturar, es urgente",
    "¿Podrían indicarme el horario de atención al cliente?",
    "Llevo tres semanas esperando el reembolso y nadie responde, es inaceptable",
    "La aplicación bancaria muestra saldos incorrectos a todos los usuarios",
    "La interfaz es confusa y lenta, cada tarea me toma el doble de tiempo",
    "Hola, muchas gracias por la atención de ayer",
]


def main() -> None:
    clf = ClasificadorTickets()
    print("LLM disponible:", clf.llm_disponible,
          "| modelo:", getattr(clf.llm, "modelo", None))
    print("-" * 90)
    for mensaje in EJEMPLOS:
        r = clf.clasifica(mensaje)
        print(f"[{r.etiqueta.upper():8s}] fuente={r.fuente:6s} prioridad={r.prioridad:5s} "
              f"-> {r.accion}")
        print(f"          «{mensaje}»")
    print("-" * 90)


if __name__ == "__main__":
    main()
