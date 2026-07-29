import "@testing-library/jest-dom/vitest";

// Locale fijo en inglés para toda la suite. La copia de la app es inglesa
// (medido: 1550 marcadores en inglés contra 53 en español, y el español estaba
// solo en el módulo de generación y en 3 pantallas de auth), así que el catálogo
// `en` es el que reproduce los textos originales VERBATIM y los tests que buscan
// texto literal son la red que detecta una extracción mal hecha: una clave mal
// escrita rinde la clave cruda y el assert falla.
localStorage.setItem("upflow.locale", "en");
