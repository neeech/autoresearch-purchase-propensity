# Issue tracker: GitHub

Repositorio: neeech/autoresearch-purchase-propensity.
Las tareas y especificaciones viven en GitHub Issues.
Usar gh desde la raíz del repositorio.

- Crear: gh issue create --title "..." --body-file <archivo>
- Consultar: gh issue view <n> --json number,title,body,labels,comments
- Listar: gh issue list --state open
- Etiquetar: gh issue edit <n> --add-label "<etiqueta>"
- Comentar: gh issue comment <n> --body-file <archivo>
- Cerrar: registrar primero el motivo y ejecutar gh issue close <n>.

Antes de publicar, verificar el destino con gh repo view.
No publicar tareas en el repositorio de las skills.

Cuando una skill indique publicar o consultar un ticket,
crear o leer una issue de este repositorio.

Usar sub-issues y dependencias nativas cuando estén disponibles.
En caso contrario, usar referencias "Part of #N" y "Blocked by #N".

PRs as a request surface: no.
