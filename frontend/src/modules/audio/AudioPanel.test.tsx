import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type {
  AudioCapabilities,
  AudioJob,
  CreateJobResponse,
  DevicesResponse,
  VoiceCatalog,
} from "../../lib/apiTypes";
import * as audioService from "../../services/audio";
import { AudioPanel } from "./AudioPanel";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});

vi.mock("../../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/audio")>();
  return {
    ...actual,
    createAudioJob: vi.fn(),
    getAudioJob: vi.fn(),
    fetchAudioCapabilities: vi.fn(),
    fetchVoiceCatalog: vi.fn(),
  };
});

const DEVICES: DevicesResponse = {
  devices: [
    { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
    { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" },
  ],
  defaultDeviceId: "dml:0",
};

function voiceStep(id: string, defaultEnabled: boolean) {
  return {
    id,
    labelKey: `voice.step.${id}.label`,
    descriptionKey: `voice.step.${id}.description`,
    kind: "filter",
    defaultEnabled,
  };
}

const VOICE_CATALOG: VoiceCatalog = {
  steps: [
    voiceStep("highpass", true),
    voiceStep("compress", true),
    voiceStep("deesser", true),
    voiceStep("loudness", false),
  ],
  deliveries: [
    {
      id: "streaming",
      labelKey: "voice.delivery.streaming.label",
      descriptionKey: "voice.delivery.streaming.description",
      lufs: -14,
      truePeakDb: -1,
    },
  ],
};

const KARAOKE_STEMS = [
  { id: "instrumental", labelKey: "audio.stem.instrumental" },
  { id: "vocals", labelKey: "audio.stem.vocals" },
];

const REVERB_STEMS = [
  { id: "dry", labelKey: "audio.stem.dry" },
  { id: "wet", labelKey: "audio.stem.wet" },
];

const DEECHO_STEMS = [
  { id: "no_echo", labelKey: "audio.stem.no_echo" },
  { id: "echo", labelKey: "audio.stem.echo" },
];

const DEREVERB_STEMS = [
  { id: "no_reverb", labelKey: "audio.stem.no_reverb" },
  { id: "reverb", labelKey: "audio.stem.reverb" },
];

const FULL_CAPABILITIES: AudioCapabilities = {
  denoiseModes: ["deepfilter", "rnnoise"],
  restoreAvailable: true,
  restoreModes: ["apollo", "audiosr"],
  masteringPresets: [
    {
      id: "streaming",
      labelKey: "audio.mastering.preset.streaming.label",
      descriptionKey: "audio.mastering.preset.streaming.description",
      targetLufs: -14,
    },
    {
      id: "voice",
      labelKey: "audio.mastering.preset.voice.label",
      descriptionKey: "audio.mastering.preset.voice.description",
      targetLufs: -16,
    },
  ],
  cleanupOverprocessingThreshold: 3,
  outputFormats: ["flac", "m4a", "mp3", "wav"],
  lossyFormats: ["m4a", "mp3"],
  lossyQualities: [
    { id: "maximum", bitrates: { mp3: "320k", m4a: "256k" } },
    { id: "balanced", bitrates: { mp3: "192k", m4a: "192k" } },
    { id: "compact", bitrates: { mp3: "128k", m4a: "128k" } },
  ],
  defaultLossyQuality: "maximum",
  cleanupSteps: [
    {
      id: "denoise",
      name: "UVR DeNoise by FoxJoy",
      family: "denoise",
      covers: ["denoise"],
      installed: true,
      descriptionKey: "audio.karaoke.model.denoise.description",
    },
    {
      id: "deecho_normal",
      name: "UVR De-Echo Normal by FoxJoy",
      family: "deecho",
      covers: ["deecho"],
      installed: true,
      descriptionKey: "audio.karaoke.model.deecho_normal.description",
    },
    {
      id: "deecho_dereverb",
      name: "UVR DeEcho-DeReverb by FoxJoy",
      family: "deecho",
      covers: ["deecho", "dereverb"],
      installed: true,
      descriptionKey: "audio.karaoke.model.deecho_dereverb.description",
    },
    {
      id: "reverb_hq",
      name: "Reverb HQ by FoxJoy",
      family: "dereverb",
      covers: ["dereverb"],
      installed: true,
      descriptionKey: "audio.karaoke.model.reverb_hq.description",
    },
  ],
  separationModels: [
    {
      id: "inst_hq_3",
      name: "MDX-Net Inst HQ 3",
      installed: true,
      primaryStem: "Instrumental",
      category: "karaoke",
      architecture: "mdx",
      descriptionKey: "audio.karaoke.model.inst_hq_3.description",
      stems: KARAOKE_STEMS,
    },
    {
      id: "voc_ft",
      name: "MDX-Net Voc FT",
      installed: false,
      primaryStem: "Vocals",
      category: "karaoke",
      architecture: "mdx",
      descriptionKey: "audio.karaoke.model.voc_ft.description",
      stems: KARAOKE_STEMS,
    },
    {
      id: "mel_band_roformer_kim",
      name: "Mel-Band RoFormer by KimberleyJSN",
      installed: true,
      primaryStem: "Vocals",
      category: "karaoke",
      architecture: "roformer",
      descriptionKey: "audio.karaoke.model.mel_band_roformer_kim.description",
      warningKey: "audio.karaoke.model.mel_band_roformer_kim.warning",
      badgeKey: "audio.karaoke.badge.slow",
      stems: KARAOKE_STEMS,
    },
    {
      id: "reverb_hq",
      name: "Reverb HQ by FoxJoy",
      installed: true,
      primaryStem: "Reverb",
      category: "cleanup",
      architecture: "mdx",
      descriptionKey: "audio.karaoke.model.reverb_hq.description",
      stems: REVERB_STEMS,
    },
    {
      id: "deecho_normal",
      name: "UVR De-Echo Normal by FoxJoy",
      installed: true,
      primaryStem: "No Echo",
      category: "cleanup",
      architecture: "vr",
      descriptionKey: "audio.karaoke.model.deecho_normal.description",
      stems: DEECHO_STEMS,
    },
    {
      id: "deecho_aggressive",
      name: "UVR De-Echo Aggressive by FoxJoy",
      installed: false,
      primaryStem: "No Echo",
      category: "cleanup",
      architecture: "vr",
      descriptionKey: "audio.karaoke.model.deecho_aggressive.description",
      stems: DEECHO_STEMS,
    },
    {
      id: "deecho_dereverb",
      name: "UVR DeEcho-DeReverb by FoxJoy",
      installed: true,
      primaryStem: "No Reverb",
      category: "cleanup",
      architecture: "vr",
      descriptionKey: "audio.karaoke.model.deecho_dereverb.description",
      stems: DEREVERB_STEMS,
    },
  ],
};

function renderPanel(capabilities: AudioCapabilities = FULL_CAPABILITIES) {
  vi.mocked(api.getDevices).mockResolvedValue(DEVICES);
  vi.mocked(audioService.fetchAudioCapabilities).mockResolvedValue(capabilities);
  vi.mocked(audioService.fetchVoiceCatalog).mockResolvedValue(VOICE_CATALOG);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<AudioPanel />, { wrapper: Wrapper });
}

function makeFile(): File {
  return new File(["binary"], "voice.wav", { type: "audio/wav" });
}

function selectFile() {
  const fileInput = document.getElementById("audio-file-input") as HTMLInputElement;
  fireEvent.change(fileInput, { target: { files: [makeFile()] } });
}

// Un archivo que YA esta en el formato de salida por defecto (FLAC): sin
// conversion posible, el envio vuelve a depender de que se elija un paso. Es la
// unica forma de probar la puerta del CTA ahora que convertir es una entrega
// valida por si sola.
function selectFileAlreadyInOutputFormat() {
  const fileInput = document.getElementById("audio-file-input") as HTMLInputElement;
  fireEvent.change(fileInput, {
    target: { files: [new File(["binary"], "voice.flac", { type: "audio/flac" })] },
  });
}

afterEach(() => {
  vi.mocked(api.getDevices).mockReset();
  vi.mocked(audioService.fetchAudioCapabilities).mockReset();
  vi.mocked(audioService.createAudioJob).mockReset();
  vi.mocked(audioService.getAudioJob).mockReset();
  vi.mocked(audioService.fetchVoiceCatalog).mockReset();
});

describe("AudioPanel", () => {
  it("renders only the denoise modes reported by capabilities", async () => {
    renderPanel({ denoiseModes: ["deepfilter"], restoreAvailable: false, restoreModes: [] });

    expect(await screen.findByRole("button", { name: "DeepFilterNet" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "RNNoise" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "None" })).toBeInTheDocument();
  });

  it("keeps the CTA disabled until a step is chosen when there is nothing to convert", async () => {
    renderPanel({ denoiseModes: ["deepfilter"], restoreAvailable: false, restoreModes: [] });
    const denoiseButton = await screen.findByRole("button", { name: "DeepFilterNet" });

    selectFileAlreadyInOutputFormat();
    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    expect(submitButton).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(/nothing to do/i);

    fireEvent.click(denoiseButton);

    await waitFor(() => expect(submitButton).not.toBeDisabled());
  });

  // El mensaje que explica por que el boton esta apagado tiene que nombrar las
  // MISMAS secciones que hay en pantalla. Nombrar una que no esta deja al
  // usuario buscando un control que no existe, y omitir una que si esta le
  // esconde una forma valida de habilitar el envio.
  it("names the mastering section as a valid choice, because choosing it alone enables the CTA", async () => {
    renderPanel(FULL_CAPABILITIES);
    // El mensaje se pinta antes de que lleguen las capabilities, y ahi todavia
    // no hay secciones opcionales: esperar a que la seccion exista es esperar al
    // estado que el mensaje tiene que describir.
    await screen.findByRole("button", { name: /^Mastering/ });

    expect(screen.getByText(/at least one/i)).toHaveTextContent(/mastering/i);
  });

  it("does not name Restore when there are no restore models installed", async () => {
    renderPanel({ denoiseModes: ["deepfilter"], restoreAvailable: false, restoreModes: [] });

    const hint = await screen.findByText(/at least one/i);

    expect(screen.queryByRole("button", { name: /^Restore/ })).not.toBeInTheDocument();
    expect(hint).not.toHaveTextContent(/restore/i);
  });

  it("shows the Apollo restore option with an Experimental badge when restore is available", async () => {
    renderPanel(FULL_CAPABILITIES);

    fireEvent.click(await screen.findByRole("button", { name: /^Restore/ }));

    expect(await screen.findByRole("button", { name: /Apollo/ })).toBeInTheDocument();
    // Apollo and AudioSR both carry the badge.
    expect(screen.getAllByText("Experimental")).toHaveLength(2);
  });

  it("offers AudioSR as a restore mode and shows its diffusion cost hint", async () => {
    renderPanel(FULL_CAPABILITIES);

    fireEvent.click(await screen.findByRole("button", { name: /^Restore/ }));
    fireEvent.click(await screen.findByRole("button", { name: /AudioSR/ }));

    expect(await screen.findByText(/per minute of audio/i)).toBeInTheDocument();
  });

  it("hides the Restore section entirely when restore is not available", async () => {
    renderPanel({ denoiseModes: ["deepfilter", "rnnoise"], restoreAvailable: false, restoreModes: [] });
    await screen.findByRole("button", { name: "DeepFilterNet" });

    expect(screen.queryByRole("button", { name: /^Restore/ })).not.toBeInTheDocument();
  });

  it("submits a job with the selected denoise, restore and device and surfaces the download link", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    };
    const completedJob: AudioJob = {
      ownerId: null,
      id: "aud-1",
      status: "completed",
      originalFilename: "voice.wav",
      denoise: "deepfilter",
      restore: "apollo",
      device: "auto",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: "2026-01-01T00:00:42Z",
      progressPct: null,
      stages: null,
      error: null,
      downloadUrl: "/api/v1/audio/jobs/aud-1/download",
    };
    vi.mocked(audioService.createAudioJob).mockResolvedValue(createResponse);
    vi.mocked(audioService.getAudioJob).mockResolvedValue(completedJob);

    renderPanel(FULL_CAPABILITIES);

    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));
    fireEvent.click(await screen.findByRole("button", { name: /^Restore/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Apollo/ }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(await screen.findByRole("link", { name: /download/i })).toHaveAttribute(
      "href",
      "/api/v1/audio/jobs/aud-1/download",
    );
    // device=null a proposito: sin eleccion explicita manda el default del
    // backend (la GPU si hay). Mandar "cpu" fijo dejaba el restore ~10x mas
    // lento en silencio (reporte de campo 2026-08-10).
    expect(vi.mocked(audioService.createAudioJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ denoise: "deepfilter", restore: "apollo", device: null }),
    );
  });

  it("advierte que la CPU es mas lenta solo cuando hay GPU y se eligio CPU", async () => {
    renderPanel(FULL_CAPABILITIES);

    fireEvent.click(await screen.findByRole("button", { name: /^Device/ }));

    // Por defecto corre en la GPU (default del backend): no hay que avisar nada.
    expect(await screen.findByRole("radio", { name: /AMD Radeon/ })).toBeChecked();
    expect(screen.queryByText(/10 veces más lento|10x slower/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: /CPU/ }));

    expect(await screen.findByText(/10 veces más lento|10x slower/i)).toBeInTheDocument();
  });

  it("shows format options with friendly descriptions and defaults to FLAC", async () => {
    renderPanel();

    expect(await screen.findByRole("radio", { name: /flac/i })).toBeChecked();
    expect(screen.getByText(/lossless.*50%|50%.*lighter|smaller/i)).toBeInTheDocument();
    expect(screen.getByText(/plays everywhere/i)).toBeInTheDocument();
    // M4A esta por compatibilidad: es el formato que pide el caso de uso real
    // (pasar un archivo al telefono), asi que tiene que estar en la lista.
    expect(screen.getByRole("radio", { name: /m4a/i })).toBeInTheDocument();
  });

  it("submits the selected output format", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "aud-2",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-2",
      downloadUrl: null,
    };
    vi.mocked(audioService.createAudioJob).mockResolvedValueOnce(createResponse);
    vi.mocked(audioService.getAudioJob).mockResolvedValue({
      ownerId: null,
      id: "aud-2",
      status: "queued",
      originalFilename: "voice.wav",
      denoise: "deepfilter",
      restore: null,
      device: null,
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      progressPct: null,
      stages: null,
      error: null,
      downloadUrl: null,
    });

    renderPanel();

    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));
    fireEvent.click(screen.getByRole("radio", { name: /^wav$/i }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    expect(vi.mocked(audioService.createAudioJob).mock.calls[0][0].outputFormat).toBe("wav");
  });

  it("lets a voice-only job through without denoise or restore", async () => {
    // Alguien que solo quiere enfocar el dialogo no tiene por que encender
    // denoise ni restore para poder enviar.
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-v", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel();
    selectFileAlreadyInOutputFormat();

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    expect(submitButton).toBeDisabled();

    // La seccion arranca colapsada: su cuerpo esta `hidden` y por lo tanto no
    // es accesible hasta abrirla.
    fireEvent.click(await screen.findByRole("button", { name: /^voice/i }));
    // El catalogo llega async: hasta que resuelve, el panel muestra el mensaje
    // de carga y no el opt-in.
    fireEvent.click(
      await screen.findByRole("checkbox", { name: /shape the voice/i }),
    );
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const sent = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(sent.denoise).toBeNull();
    expect(sent.restore).toBeNull();
    expect(sent.voiceSteps?.sort()).toEqual(["compress", "deesser", "highpass"]);
  });

  it("sends no voice fields while the voice section is off", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-n", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const sent = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(sent.voiceSteps).toEqual([]);
    expect(sent.voiceDelivery).toBeNull();
    expect(sent.voicePresenceDb).toBeNull();
  });


  it("offers the mastering presets and sends the chosen one", async () => {
    // Nivelar el volumen a un estandar es lo que separa un audio procesado de uno
    // entregable: dos archivos tratados igual sonaban a distinto volumen.
    const submitSpy = vi.mocked(audioService.createAudioJob);
    submitSpy.mockResolvedValue({ jobId: "a1", status: "queued", statusUrl: "", downloadUrl: null });
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: /^Mastering/ }));

    selectFile();
    fireEvent.click(screen.getByRole("button", { name: "Streaming" }));
    fireEvent.click(screen.getByRole("button", { name: /enhance audio/i }));

    await waitFor(() => expect(submitSpy).toHaveBeenCalled());
    expect(submitSpy.mock.calls[0][0].master).toBe("streaming");
  });

  it("lets levelling be the only thing asked for", async () => {
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: /^Mastering/ }));

    selectFile();
    fireEvent.click(screen.getByRole("button", { name: "Streaming" }));

    expect(screen.getByRole("button", { name: /enhance audio/i })).toBeEnabled();
  });
});

/**
 * El cuerpo de una AccordionSection, para acotar las aserciones a SU seccion.
 * Hace falta desde que los modelos de limpieza aparecen en dos lugares: el
 * picker de Karaoke (una pasada, dos stems) y la cadena de Limpieza.
 */
function bodyOf(header: HTMLElement): HTMLElement {
  const id = header.getAttribute("aria-controls");
  return document.getElementById(id ?? "") as HTMLElement;
}

describe("AudioPanel cadena de limpieza", () => {
  it("submits the cleanup chain together with the rest of the chain", async () => {
    // La limpieza dejó de ser exclusiva: convive con mastering en el MISMO job.
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-c", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel();
    selectFile();

    fireEvent.click(
      await screen.findByRole("checkbox", { name: /clean up the recording/i }),
    );
    fireEvent.click(screen.getByRole("checkbox", { name: /UVR DeNoise/i }));
    fireEvent.click(await screen.findByRole("button", { name: /^mastering/i }));
    fireEvent.click(screen.getByRole("button", { name: "Streaming" }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const sent = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(sent.cleanupSteps).toEqual(["denoise"]);
    expect(sent.master).toBe("streaming");
    expect(sent.separate).toBeUndefined();
  });

  it("lets a cleanup-only job through without denoise or restore", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-c2", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel();
    selectFileAlreadyInOutputFormat();

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    expect(submitButton).toBeDisabled();

    fireEvent.click(
      await screen.findByRole("checkbox", { name: /clean up the recording/i }),
    );
    fireEvent.click(screen.getByRole("checkbox", { name: /UVR De-Echo Normal/i }));
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const sent = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(sent.denoise).toBeNull();
    expect(sent.restore).toBeNull();
    expect(sent.cleanupSteps).toEqual(["deecho_normal"]);
  });

  it("sends the chain in catalog order no matter the ticking order", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-c3", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel();
    selectFile();

    fireEvent.click(
      await screen.findByRole("checkbox", { name: /clean up the recording/i }),
    );
    fireEvent.click(screen.getByRole("checkbox", { name: /Reverb HQ/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /UVR DeNoise/i }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const sent = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(sent.cleanupSteps).toEqual(["denoise", "reverb_hq"]);
  });

  it("sends no cleanup steps while the section is off", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-c4", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    expect(vi.mocked(audioService.createAudioJob).mock.calls[0][0].cleanupSteps).toEqual([]);
  });

  it("says the noise reduction section is for voice, not for music", async () => {
    // La confusión reportada: "Reducción de ruido" no decía en ningún lado que
    // sus modelos están entrenados con habla y en música apagan instrumentos.
    renderPanel();

    expect(
      await screen.findByText(/These models learned to recognise speech/i),
    ).toBeInTheDocument();
    // El titulo de la seccion tambien lo nombra: la etiqueta dejo de ser un
    // "quitar ruido" generico que se lee como valido para cualquier material.
    expect(
      screen.getByRole("button", { name: /^Noise reduction \(voice\)/ }),
    ).toBeInTheDocument();
  });

  it("describes the chosen denoise mode, naming its material", async () => {
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));

    expect(
      await screen.findByText(/the stronger of the two.*in music it can mute instruments/i),
    ).toBeInTheDocument();
  });

  it("hides the cleanup section when the backend reports no catalog", async () => {
    // Misma política que Acabado y Restauración: la sección existe cuando el
    // backend reporta su catálogo, no antes.
    renderPanel({ denoiseModes: ["rnnoise"], restoreAvailable: false, restoreModes: [] });

    await screen.findByRole("button", { name: "RNNoise" });
    expect(screen.queryByRole("button", { name: /^cleanup/i })).not.toBeInTheDocument();
  });
});

describe("AudioPanel modo karaoke", () => {
  async function openKaraoke() {
    fireEvent.click(await screen.findByRole("button", { name: /^Karaoke/ }));
    return screen.findByRole("checkbox", { name: /split the audio into two stems/i });
  }

  it("submits a separate-only job with the installed model and nulls the rest", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "kar-1", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel(FULL_CAPABILITIES);
    selectFile();
    // Aunque haya un denoise elegido de antes, activar karaoke lo excluye del
    // form: el backend rechaza combinarlos.
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));
    fireEvent.click(await openKaraoke());

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const sent = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(sent.separate).toBe(true);
    expect(sent.separationModel).toBe("inst_hq_3");
    expect(sent.denoise).toBeNull();
    expect(sent.restore).toBeNull();
    expect(sent.master).toBeNull();
    expect(sent.voiceSteps).toEqual([]);
  });

  it("explains why the other sections are disabled while karaoke is on", async () => {
    renderPanel(FULL_CAPABILITIES);
    fireEvent.click(await openKaraoke());

    expect(await screen.findByText(/karaoke runs alone/i)).toBeInTheDocument();
  });

  it("marks the attenuated sections wrapper as inert while karaoke is on", async () => {
    // pointer-events-none solo bloquea el mouse: sin inert, Tab + Enter seguian
    // operando los controles atenuados y la seleccion se descartaba en silencio.
    renderPanel(FULL_CAPABILITIES);
    const denoiseButton = await screen.findByRole("button", { name: "DeepFilterNet" });
    const wrapper = denoiseButton.closest("[aria-disabled]");
    expect(wrapper).not.toBeNull();
    expect(wrapper).not.toHaveAttribute("inert");

    fireEvent.click(await openKaraoke());

    expect(wrapper).toHaveAttribute("inert");
    expect(wrapper).toHaveClass("pointer-events-none", "opacity-40");
  });

  it("offers a download button for models that are not installed yet", async () => {
    renderPanel(FULL_CAPABILITIES);
    await openKaraoke();

    expect(screen.getByText(/MDX-Net Voc FT.*not downloaded/i)).toBeInTheDocument();
    expect(
      screen.getByText(/UVR De-Echo Aggressive by FoxJoy.*not downloaded/i),
    ).toBeInTheDocument();
    // Una tarjeta de descarga por cada modelo que falta, de la arquitectura
    // que sea: el usuario no elige arquitecturas, elige modelos.
    expect(screen.getAllByRole("button", { name: /download/i })).toHaveLength(2);
    // El instalado es un boton de seleccion, no una tarjeta de descarga.
    expect(screen.getByRole("button", { name: "MDX-Net Inst HQ 3" })).toBeInTheDocument();
  });

  it("lists the three cleanup passes and tells them apart", async () => {
    renderPanel(FULL_CAPABILITIES);
    fireEvent.click(await openKaraoke());

    // Los tres De-Echo caen en Limpieza junto a Reverb HQ, en UNA sola lista.
    // La leyenda dice ademas la FORMA de la salida: aca cada modelo corre solo
    // y devuelve dos stems, a diferencia de la seccion Limpieza encadenable.
    expect(screen.getByText("Cleanup (one pass, two stems)")).toBeInTheDocument();
    const karaoke = within(bodyOf(screen.getByRole("button", { name: /^Karaoke/ })));
    fireEvent.click(karaoke.getByRole("button", { name: "UVR De-Echo Normal by FoxJoy" }));
    expect(await karaoke.findByText(/moderate echo/i)).toBeInTheDocument();
    expect(screen.getByText("No echo + Echo")).toBeInTheDocument();

    fireEvent.click(karaoke.getByRole("button", { name: "UVR DeEcho-DeReverb by FoxJoy" }));
    expect(await karaoke.findByText(/echo AND room reverb/i)).toBeInTheDocument();
    expect(screen.getByText("No echo or reverb + Echo and reverb")).toBeInTheDocument();
  });

  it("submits the selected de-echo model", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "de-1",
      status: "queued",
      statusUrl: "/x",
      downloadUrl: null,
    });
    renderPanel(FULL_CAPABILITIES);
    selectFile();
    fireEvent.click(await openKaraoke());
    fireEvent.click(screen.getByRole("button", { name: "UVR De-Echo Normal by FoxJoy" }));
    fireEvent.click(screen.getByRole("button", { name: /enhance audio/i }));

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const payload = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(payload.separate).toBe(true);
    expect(payload.separationModel).toBe("deecho_normal");
  });

  it("keeps the CTA disabled when karaoke is on but no model is installed", async () => {
    renderPanel({
      denoiseModes: [],
      restoreAvailable: false,
      restoreModes: [],
      separationModels: [
        {
          id: "inst_hq_3",
          name: "MDX-Net Inst HQ 3",
          installed: false,
          primaryStem: "Instrumental",
          category: "karaoke",
          architecture: "mdx",
          descriptionKey: "audio.karaoke.model.inst_hq_3.description",
          stems: KARAOKE_STEMS,
        },
      ],
    });
    selectFile();
    fireEvent.click(await openKaraoke());

    expect(await screen.findByText(/at least one separation model/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /enhance audio/i })).toBeDisabled();
  });

  it("shows both stem download links when a karaoke job completes", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "kar-2", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    vi.mocked(audioService.getAudioJob).mockResolvedValue({
      ownerId: null,
      id: "kar-2",
      status: "completed",
      originalFilename: "song.wav",
      denoise: null,
      restore: null,
      device: "cpu",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: "2026-01-01T00:02:00Z",
      progressPct: null,
      stages: null,
      error: null,
      separate: true,
      separationModel: "inst_hq_3",
      downloadUrl: "/api/v1/audio/jobs/kar-2/download",
      vocalsDownloadUrl: "/api/v1/audio/jobs/kar-2/download?stem=vocals",
    });
    renderPanel(FULL_CAPABILITIES);
    selectFile();
    fireEvent.click(await openKaraoke());

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(await screen.findByRole("link", { name: /instrumental/i })).toHaveAttribute(
      "href",
      "/api/v1/audio/jobs/kar-2/download",
    );
    expect(screen.getByRole("link", { name: /vocals/i })).toHaveAttribute(
      "href",
      "/api/v1/audio/jobs/kar-2/download?stem=vocals",
    );
  });

  it("credits UVR in the karaoke section", async () => {
    renderPanel(FULL_CAPABILITIES);
    await openKaraoke();

    expect(screen.getByText(/ultimate vocal remover/i)).toBeInTheDocument();
  });

  it("warns that the max-quality model is slow BEFORE it is selected", async () => {
    // El punto entero de la advertencia: enterarse del ~9x cuando el trabajo
    // ya arrancó no es haber elegido. Con inst_hq_3 seleccionado (el default),
    // el texto del carril lento ya tiene que estar en pantalla.
    renderPanel(FULL_CAPABILITIES);
    fireEvent.click(await openKaraoke());

    const warning = await screen.findByText(/20x slower than Inst HQ 3/i);
    expect(warning).toBeInTheDocument();
    expect(warning).toHaveTextContent("Mel-Band RoFormer by KimberleyJSN");
    // Y no es que esté seleccionado: el elegido sigue siendo el default.
    expect(
      screen.getByRole("button", { name: /MDX-Net Inst HQ 3/ }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("button", { name: /Mel-Band RoFormer/ }),
    ).toHaveAttribute("aria-pressed", "false");
  });

  it("shows each model's own badge, not one hardcoded label", async () => {
    // La insignia sale de `badgeKey` y no de "tiene warningKey". Mientras hubo un
    // solo modelo con advertencia —el lento— dibujar "Slow" fijo funcionaba de
    // casualidad; con el segundo, el catalogo empezo a llamar lento al modelo mas
    // rapido de los dos.
    renderPanel(FULL_CAPABILITIES);
    fireEvent.click(await openKaraoke());

    expect(
      await screen.findByRole("button", { name: /Mel-Band RoFormer by KimberleyJSN Slow/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /MDX-Net Inst HQ 3/ }),
    ).not.toHaveTextContent("Slow");
  });

  it("shows no badge for a model that warns without declaring one", async () => {
    // La regresion concreta: si la insignia volviera a colgar de `warningKey`,
    // este modelo mostraria un texto que nadie eligio para el.
    const sinInsignia = {
      ...FULL_CAPABILITIES,
      separationModels: FULL_CAPABILITIES.separationModels.map((model) =>
        model.id === "mel_band_roformer_kim"
          ? { ...model, badgeKey: null }
          : model,
      ),
    };
    renderPanel(sinInsignia);
    fireEvent.click(await openKaraoke());

    expect(
      await screen.findByRole("button", { name: /Mel-Band RoFormer by KimberleyJSN/ }),
    ).not.toHaveTextContent("Slow");
  });

  it("submits the roformer model when it is picked", async () => {
    renderPanel(FULL_CAPABILITIES);
    fireEvent.click(await openKaraoke());
    fireEvent.click(
      screen.getByRole("button", { name: /Mel-Band RoFormer by KimberleyJSN/ }),
    );

    expect(
      await screen.findByText(/highest quality in the catalog/i),
    ).toBeInTheDocument();
  });

  it("groups the picker by category and describes the cleanup model", async () => {
    renderPanel(FULL_CAPABILITIES);
    fireEvent.click(await openKaraoke());

    // Dos grupos: los modelos karaoke y la pasada de limpieza, separados.
    expect(screen.getByText("Cleanup (one pass, two stems)")).toBeInTheDocument();
    const karaoke = within(bodyOf(screen.getByRole("button", { name: /^Karaoke/ })));
    fireEvent.click(karaoke.getByRole("button", { name: "Reverb HQ by FoxJoy" }));

    expect(
      await karaoke.findByText(/second pass to clean up the instrumental/i),
    ).toBeInTheDocument();
    // El resumen de la sección nombra los stems del modelo elegido.
    expect(screen.getByText("No reverb (dry) + Reverb (wet)")).toBeInTheDocument();
  });

  it("submits the cleanup model and labels the stem downloads from the catalog", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "rev-1", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    vi.mocked(audioService.getAudioJob).mockResolvedValue({
      ownerId: null,
      id: "rev-1",
      status: "completed",
      originalFilename: "instrumental.flac",
      denoise: null,
      restore: null,
      device: "cpu",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: "2026-01-01T00:02:00Z",
      progressPct: null,
      stages: null,
      error: null,
      separate: true,
      separationModel: "reverb_hq",
      downloadUrl: "/api/v1/audio/jobs/rev-1/download",
      stems: [
        {
          id: "dry",
          labelKey: "audio.stem.dry",
          url: "/api/v1/audio/jobs/rev-1/download?stem=dry",
        },
        {
          id: "wet",
          labelKey: "audio.stem.wet",
          url: "/api/v1/audio/jobs/rev-1/download?stem=wet",
        },
      ],
      vocalsDownloadUrl: null,
    });
    renderPanel(FULL_CAPABILITIES);
    selectFile();
    fireEvent.click(await openKaraoke());
    fireEvent.click(screen.getByRole("button", { name: "Reverb HQ by FoxJoy" }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    expect(vi.mocked(audioService.createAudioJob).mock.calls[0][0].separationModel).toBe(
      "reverb_hq",
    );
    // "Sin reverb" primero: es el stem que el usuario quiere.
    const dryLink = await screen.findByRole("link", { name: /no reverb \(dry\)/i });
    expect(dryLink).toHaveAttribute("href", "/api/v1/audio/jobs/rev-1/download?stem=dry");
    const wetLink = screen.getByRole("link", { name: /reverb \(wet\)/i });
    expect(wetLink).toHaveAttribute("href", "/api/v1/audio/jobs/rev-1/download?stem=wet");
    expect(dryLink.compareDocumentPosition(wetLink) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});

describe("AudioPanel en lote", () => {
  it("creates one job per file with the same settings", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    });
    renderPanel(FULL_CAPABILITIES);

    const fileInput = document.getElementById("audio-file-input") as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: {
        files: [
          new File(["a"], "uno.wav", { type: "audio/wav" }),
          new File(["b"], "dos.wav", { type: "audio/wav" }),
        ],
      },
    });
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() =>
      expect(vi.mocked(audioService.createAudioJob)).toHaveBeenCalledTimes(2),
    );
    const enviados = vi
      .mocked(audioService.createAudioJob)
      .mock.calls.map((call) => call[0]);
    expect(enviados.map((p) => p.file.name)).toEqual(["uno.wav", "dos.wav"]);
    expect(enviados[1].denoise).toBe(enviados[0].denoise);
  });
});

// ---------------------------------------------------------------------------
// Conversion de formato sin procesar nada. El caso real: alguien con un FLAC
// que necesita un MP3 por compatibilidad no tiene ningun paso que pedir, y
// hasta ahora el boton lo dejaba afuera pidiendole que eligiera una seccion.
// ---------------------------------------------------------------------------

describe("AudioPanel conversion de formato", () => {
  function selectFlacFile() {
    const fileInput = document.getElementById("audio-file-input") as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: { files: [new File(["b"], "song.flac", { type: "audio/flac" })] },
    });
  }

  function pickFormat(label: RegExp) {
    fireEvent.click(screen.getByRole("radio", { name: label }));
  }

  it("says up front that converting works without picking any step", async () => {
    renderPanel();

    expect(
      await screen.findByText(/conversion works on its own|converted in a single pass/i),
    ).toBeInTheDocument();
  });

  it("enables the CTA with only a file and a different output format", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });

    selectFlacFile();
    pickFormat(/mp3/i);

    const submitButton = screen.getByRole("button", { name: /convert/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
  });

  it("renames the CTA to Convert while no step is selected", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });

    selectFlacFile();
    pickFormat(/mp3/i);

    expect(screen.getByRole("button", { name: /^convert$/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /enhance audio/i })).not.toBeInTheDocument();
  });

  it("goes back to the enhance CTA as soon as a step is picked", async () => {
    renderPanel();
    selectFlacFile();
    pickFormat(/mp3/i);

    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));

    expect(await screen.findByRole("button", { name: /enhance audio/i })).toBeInTheDocument();
  });

  it("warns that lossless to lossy cannot be undone, without blocking it", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });

    selectFlacFile();
    pickFormat(/mp3/i);

    expect(screen.getByText(/cannot be recovered|discards audio data/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /convert/i })).not.toBeDisabled();
  });

  it("does not warn when the target keeps the audio lossless", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });

    selectFlacFile();
    pickFormat(/wav/i);

    expect(screen.queryByText(/cannot be recovered|discards audio data/i)).not.toBeInTheDocument();
  });

  it("offers the quality tiers only for lossy targets", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });

    expect(screen.queryByRole("radio", { name: /maximum/i })).not.toBeInTheDocument();

    pickFormat(/mp3/i);

    expect(await screen.findByRole("radio", { name: /maximum/i })).toBeChecked();
    expect(screen.getByText(/320kbps/i)).toBeInTheDocument();
  });

  it("shows the bitrate of the format actually selected", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });

    pickFormat(/m4a/i);

    // 256k es el bitrate de AAC en el escalon maximo; 320k es el de MP3.
    expect(await screen.findByText(/256kbps/i)).toBeInTheDocument();
    expect(screen.queryByText(/320kbps/i)).not.toBeInTheDocument();
  });

  it("submits a conversion-only job with no processing step and the chosen quality", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-conv", status: "queued", statusUrl: "/x", downloadUrl: null,
    });
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });
    selectFlacFile();
    pickFormat(/mp3/i);
    fireEvent.click(await screen.findByRole("radio", { name: /compact/i }));

    fireEvent.click(screen.getByRole("button", { name: /convert/i }));

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    const sent = vi.mocked(audioService.createAudioJob).mock.calls[0][0];
    expect(sent.outputFormat).toBe("mp3");
    expect(sent.lossyQuality).toBe("compact");
    expect(sent.denoise).toBeNull();
    expect(sent.restore).toBeNull();
    expect(sent.master).toBeNull();
    expect(sent.cleanupSteps).toEqual([]);
  });

  it("explains that a file already in the output format has nothing to do", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /flac/i });

    selectFlacFile();

    expect(screen.getByText(/nothing to do/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /enhance audio/i })).toBeDisabled();
  });
});
