import "@testing-library/jest-dom/vitest";

// Locale fijo en español para toda la suite. La app se escribió en español, así
// que el catálogo `es` reproduce los textos originales VERBATIM: los tests que
// buscan texto literal son entonces la red que detecta una extracción mal hecha
// (una clave mal escrita rinde la clave cruda y el assert falla). Sin fijarlo,
// jsdom reporta navigator.language "en-US" y la suite correría en inglés.
localStorage.setItem("upflow.locale", "es");
