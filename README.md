# Prueba Técnica – Data Science (Clustering con SQLite y Python)
Esta prueba tiene como objetivo evaluar tus habilidades en análisis de datos, preprocesamiento, clustering y generación de insights a partir de una base de datos en SQLite. Además, se valorará la correcta estructuración del código para asegurar su integración en un sistema más amplio, priorizando buenas prácticas de desarrollo y bajo acoplamiento.

# Descripción del Problema
Se te proporciona una base de datos en SQLite (db_test.sqlite3) que contiene información sobre términos de búsqueda, su rendimiento y la cuenta que generó esta información.

Tu tarea es:

1. **Extraer** los datos correspondientes al cliente con cuenta '5555555555'.
2. **Analizar** la información y determinar qué conclusiones pueden extraerse a través de un proceso de clustering.
3. **Agrupar** los datos mediante clustering y asignar etiquetas a cada grupo. La cantidad de clusters queda a tu criterio, pero debe ser de al menos tres (3).
4. **Guardar** los resultados del clustering en una nueva tabla llamada "clusters"

# Diseño e Implementación
El código debe seguir buenas prácticas de desarrollo, priorizando la **modularidad** y un **bajo acoplamiento** entre las etapas de extracción y procesamiento de datos.

Para gestionar la cuenta del cliente, se deberá utilizar un dataclass, obteniendo los datos de las cuentas desde el siguiente endpoint REST:

🔗 https://my-json-server.typicode.com/Ek-Adan/fake_account_api/accounts

```python
@dataclass
class AdsConfigData:
    name:str
    account_id:str
```

Se recomienda, aunque no es obligatorio, el uso de Programación Orientada a Objetos (POO) para estructurar la solución de manera más organizada y escalable.

### Entrega:
Asegúrate de incluir en el repositorio todo lo necesario para ejecutar tu solución sin inconvenientes.
