# Laboratorio Daçel — Guía paso a paso

Resultado final: una página en **https://dacel.edazelfernandez.com** que se actualiza sola
el día 1 de cada mes, y un botón en tu sitio principal que lleva a ella.

Tiempo estimado: 30 a 45 minutos la primera vez. Costo: 0 pesos.

---

## Paso 1. Crear la cuenta de GitHub
1. Entra a https://github.com y crea una cuenta gratuita.
2. Anota tu nombre de usuario. En esta guía aparece como `TUUSUARIO`.

## Paso 2. Crear el repositorio
1. Arriba a la derecha: **+** → **New repository**.
2. Nombre: `laboratorio-dacel`.
3. Marca **Public** (GitHub Pages gratis requiere que sea público).
4. Clic en **Create repository**.

## Paso 3. Subir los archivos
1. Descomprime el .zip en tu computadora.
2. En el repositorio, clic en **uploading an existing file**.
3. Arrastra TODO el contenido de la carpeta: `actualizar_datos.py`, `dacel_validacion.py`,
   `config_dacel.json`, `requirements.txt`, `plantilla_datos.csv`, `LEEME.md`,
   esta guía y la carpeta `docs`.
4. Clic en **Commit changes**.

**Importante:** la carpeta `.github` empieza con punto y muchas computadoras la ocultan,
así que casi siempre NO se sube arrastrando. Créala a mano:
1. **Add file** → **Create new file**.
2. En el nombre escribe exactamente: `.github/workflows/actualizar.yml`
3. Abre ese archivo del .zip con el Bloc de notas, copia todo y pégalo.
4. **Commit changes**.

Revisa también que exista `docs/CNAME` y `docs/.nojekyll`. Si no se subieron, créalos igual:
`docs/CNAME` con el texto `dacel.edazelfernandez.com`, y `docs/.nojekyll` vacío.

## Paso 4. Dar permiso al robot
1. **Settings** → **Actions** → **General**.
2. Hasta abajo, en **Workflow permissions**, elige **Read and write permissions**.
3. **Save**.

## Paso 5. Correr el laboratorio por primera vez
1. Pestaña **Actions** → **Actualizar Laboratorio Daçel** → **Run workflow** → **Run workflow**.
2. Espera 2 a 4 minutos hasta ver la palomita verde.
3. Si sale una X roja, abre la corrida, copia el mensaje de error y compártemelo.

## Paso 6. Publicar la página
1. **Settings** → **Pages**.
2. En **Source**: **Deploy from a branch**. Branch: **main**. Carpeta: **/docs**. **Save**.
3. En **Custom domain** debe aparecer `dacel.edazelfernandez.com`. Si no, escríbelo y guarda.

## Paso 7. Conectar el subdominio en Namecheap
1. Namecheap → **Domain List** → `edazelfernandez.com` → **Manage** → **Advanced DNS**.
2. **Add New Record**:
   - Type: **CNAME Record**
   - Host: `dacel`
   - Value: `TUUSUARIO.github.io.`
   - TTL: Automatic
3. Guarda con la palomita verde.
4. No toques los registros de `www` ni de `corporacion`: siguen apuntando a ChatGPT Sites.
5. Espera de 15 minutos a unas horas. En GitHub **Settings → Pages** aparecerá
   "DNS check successful". Entonces activa **Enforce HTTPS**.

## Paso 8. Enlazarlo desde tu sitio
En la sección de Teoría Daçel de tu página en ChatGPT Sites, agrega un botón:
- Texto: **Ver el laboratorio en vivo**
- Enlace: `https://dacel.edazelfernandez.com`

Si ChatGPT Sites te permite insertar código HTML, también puedes mostrarlo dentro de la página:

    <iframe src="https://dacel.edazelfernandez.com" width="100%" height="1400"
            style="border:0" title="Laboratorio Daçel"></iframe>

Si no lo permite, el botón funciona igual de bien.

## Paso 9. Google
- Analytics ya está incluido en el tablero con tu identificador G-BLMWEQFNX1.
- En Search Console: si tu propiedad es de tipo **Dominio**, el subdominio ya está cubierto.
  Si es de tipo **Prefijo de URL**, agrega `https://dacel.edazelfernandez.com` como propiedad nueva.

---

## Mantenimiento
- **No hay que hacer nada**: el día 1 de cada mes corre solo.
- **Si una fuente falla**, la tabla "Fuentes de datos" del tablero lo muestra en rojo y se usan
  los datos guardados del mes anterior. Para corregirla, edita `config_dacel.json`.
- **Regla de oro**: si cambias una variable, un peso o un año de perturbación, escríbelo en
  `"cambios"` dentro de `config_dacel.json` con la fecha y el motivo. Se publica en el tablero.
  Así nadie puede decir que se ajustó la teoría para que "saliera".

## Qué esperar en la primera corrida
Varias pruebas saldrán **No concluyente**. Es normal: las series mundiales de supercómputo,
artículos científicos y salud mental empiezan entre 1990 y 2000, así que hay pocos años
completos. El tablero lo explica, y cada año nuevo de datos acerca un veredicto real.
