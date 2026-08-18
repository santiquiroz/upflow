/** Varias URLs pegadas de una: una por línea, en el orden en que las pegó.
 *
 * El campo sigue siendo uno solo porque el caso normal es una sola URL, con su
 * vista previa. Pegar una lista es lo que pasa cuando alguien copia media
 * playlist a mano, y hasta ahora obligaba a repetir el proceso de a uno.
 *
 * Se limpian líneas vacías y espacios: copiar de un chat o de un bloc de notas
 * arrastra basura, y una línea en blanco no es una descarga.
 */
export function parseUrlList(texto: string): string[] {
  return texto
    .split(/[\r\n]+/)
    .map((linea) => linea.trim())
    .filter((linea) => linea.length > 0);
}

export function isMultiUrl(texto: string): boolean {
  return parseUrlList(texto).length > 1;
}
