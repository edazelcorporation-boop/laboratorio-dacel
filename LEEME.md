# Laboratorio Daçel (v2.0)

Sistema automático que pone a prueba la Teoría Daçel de Edazel Fernández con datos públicos
mundiales y publica los resultados en https://dacel.edazelfernandez.com

**Para instalarlo, sigue `GUIA_PASO_A_PASO.md`.**

## Archivos
- `actualizar_datos.py`: descarga los datos (Banco Mundial, Our World in Data).
- `dacel_validacion.py`: calcula CIDI y HDF y corre las pruebas H1–H4 y la Ecuación General.
- `config_dacel.json`: fuentes, años de perturbación y registro de cambios.
- `.github/workflows/actualizar.yml`: el robot que lo corre cada mes.
- `docs/index.html`: el tablero público.
- `plantilla_datos.csv`: para correrlo a mano con tus propios datos.

## Variables usadas
| Símbolo | Variable real | Fuente |
|---|---|---|
| E | Consumo mundial de energía primaria | Our World in Data |
| I | Potencia del supercomputador más rápido | Our World in Data (TOP500) |
| X | Suscripciones móviles por 100 personas | Banco Mundial |
| K | Artículos científicos publicados | Banco Mundial |
| A | Prevalencia de ansiedad | Our World in Data (IHME) |
| Dp | Prevalencia de depresión | Our World in Data (IHME) |
| conexion | Usuarios de internet (%) | Banco Mundial |

No existe una serie mundial anual de soledad (S) ni de dependencia digital (D); el programa
funciona sin ellas y las usará si algún día se agregan.

## Uso manual (opcional)
    pip install -r requirements.txt
    python dacel_validacion.py --demo
    python actualizar_datos.py
    python dacel_validacion.py --datos datos/datos_mundo.csv --salida docs/resultados

El modo `--demo` usa dos mundos sintéticos (uno con mecanismo Daçel y otro sin él) para
comprobar que las pruebas distinguen. H2, H3 y la Ecuación General distinguen; H1 pasa en
ambos mundos, así que por sí sola no respalda a Daçel.
