import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip().rstrip("/")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").strip().rstrip("/")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash").strip()

MAX_IMAGE_BYTES = 8 * 1024 * 1024

PROMPT = """
Sei un estrattore di turni per un'app personale di un autista.
Analizza lo screenshot e restituisci solo JSON valido.

Regole:
- Estrai la data del turno nel formato YYYY-MM-DD.
- Gli orari principali del turno NON vanno presi dalle righe servizio se in alto esiste la testata del turno.
- Nella testata superiore del turno, subito dopo il giorno/data, ci sono due orari principali:
  - il primo orario a sinistra, evidenziato in giallo/arancio, e' l'inizio turno
  - il secondo orario a destra, evidenziato in verde, e' la fine turno
- Se questi due orari superiori sono visibili, sono loro i valori corretti di startTime ed endTime del turno completo.
- Regola importante per i turni con piu' righe: quando a sinistra si ripete la stessa coppia linea-treno su piu' righe consecutive, considera quel blocco come un unico servizio.
- In quel caso:
  - startTime del blocco = primo orario in alto a sinistra del blocco
  - endTime del blocco = ultimo orario in basso a destra del blocco
- Non usare gli orari intermedi delle righe ripetute come fine del turno, se esiste un ultimo orario piu' in basso a destra nello stesso blocco linea-treno.
- Estrai i segmenti di servizio reali nell'ordine cronologico.
- Un segmento cambia quando cambia linea oppure treno.
- Per ogni segmento restituisci:
  - startTime: orario inizio reale del segmento nel formato HH:MM
  - endTime: orario fine reale del segmento nel formato HH:MM
  - line: linea servizio, ad esempio 09. Usa CV solo se non c'e' davvero una linea numerica migliore.
  - train: solo numero del treno, ad esempio 06 oppure 21. Non scrivere mai linea-treno insieme come 09-06.
  - vehicle: codice turno/veicolo grande mostrato nello screenshot, ad esempio C412 o CD14
- Se c'e' un'attivita' separata come TG o DISP fuori dalla linea, puoi restituirla come segmento separato:
  - con line vuota
  - con train vuoto
  - con vehicle valorizzato a TG oppure DISP
- Non usare come vehicle/turno automatico questi valori: TG, TG1, TG2, TG3, TG4, TG5, TG6, TG7, TG8, TG9, TG10, NOL, CV.
- Ignora metriche riepilogative come Nastro, Lavoro, Guida, Dur, Brk.
- Ignora le durate tipo 07.14 o 7:01 se non sono orari reali di inizio/fine servizio.
- I blocchi DISP non sono un segmento principale di linea/treno, ma possono essere attivita' separate se hanno un orario reale.
- Se c'e' un solo segmento restituisci un array con un elemento.
- Se sei incerto, fai la migliore stima possibile e spiega il dubbio in notes.

Rispondi solo con questo JSON:
{
  "date": "YYYY-MM-DD",
  "segments": [
    {
      "startTime": "HH:MM",
      "endTime": "HH:MM",
      "line": "",
      "train": "",
      "vehicle": ""
    }
  ],
  "notes": ""
}
""".strip()


class ShiftSegment(BaseModel):
    startTime: str
    endTime: str
    line: str = ""
    train: str = ""
    vehicle: str = ""


class ShiftAnalysis(BaseModel):
    date: str
    segments: list[ShiftSegment]
    notes: str = ""


app = FastAPI(title="BusTime AI Backend")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/provider")
async def provider_status() -> dict[str, str]:
    return {"provider": active_provider()}


@app.post("/analyze-shift-image")
async def analyze_shift_image(file: UploadFile = File(...)) -> JSONResponse:
    provider = active_provider()
    if provider == "none":
        raise api_error(
            status_code=500,
            code="missing_server_key",
            message="Configura GEMINI_API_KEY oppure OPENAI_API_KEY sul server.",
        )
    if not file.content_type or not file.content_type.startswith("image/"):
        raise api_error(
            status_code=400,
            code="invalid_file_type",
            message="Carica un file immagine valido.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise api_error(
            status_code=400,
            code="empty_file",
            message="Il file immagine e' vuoto.",
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise api_error(
            status_code=413,
            code="image_too_large",
            message="Immagine troppo grande. Limite 8 MB.",
        )

    if provider == "gemini":
        parsed = await analyze_with_gemini(file.content_type, image_bytes)
        model_used = GEMINI_VISION_MODEL
    else:
        parsed = await analyze_with_openai(file.content_type, image_bytes)
        model_used = OPENAI_VISION_MODEL

    summary = build_summary(parsed)
    return JSONResponse(
        {
            "date": parsed.date,
            "segments": [segment.model_dump() for segment in parsed.segments],
            "notes": parsed.notes,
            "summary": summary,
            "model": model_used,
            "provider": provider,
        }
    )


def active_provider() -> str:
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "none"


async def analyze_with_openai(content_type: str, image_bytes: bytes) -> ShiftAnalysis:
    data_url = build_data_url(content_type, image_bytes)
    payload = {
        "model": OPENAI_VISION_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OPENAI_API_BASE}/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        raise map_openai_error(response)

    output_text = extract_openai_output_text(response.json())
    return parse_analysis(output_text)


async def analyze_with_gemini(content_type: str, image_bytes: bytes) -> ShiftAnalysis:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT},
                    {
                        "inline_data": {
                            "mime_type": content_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{GEMINI_API_BASE}/models/{GEMINI_VISION_MODEL}:generateContent",
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json=payload,
        )

    if response.status_code >= 400:
        raise map_gemini_error(response)

    output_text = extract_gemini_output_text(response.json())
    return parse_analysis(output_text)


def parse_analysis(output_text: str) -> ShiftAnalysis:
    try:
        parsed_json = json.loads(output_text)
        parsed = ShiftAnalysis.model_validate(parsed_json)
    except Exception as exc:
        raise api_error(
            status_code=502,
            code="invalid_ai_response",
            message=f"Risposta AI non valida: {exc}",
        ) from exc

    if not parsed.segments:
        raise api_error(
            status_code=502,
            code="invalid_ai_response",
            message="L'AI non ha restituito segmenti validi.",
        )
    return parsed


def build_data_url(content_type: str, image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def extract_openai_output_text(response_json: dict) -> str:
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    raise api_error(502, "invalid_ai_response", "Nessun output testuale trovato nella risposta OpenAI.")


def extract_gemini_output_text(response_json: dict) -> str:
    candidates = response_json.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            text = part.get("text")
            if text:
                return text
    raise api_error(502, "invalid_ai_response", "Nessun output testuale trovato nella risposta Gemini.")


def build_summary(parsed: ShiftAnalysis) -> str:
    first = parsed.segments[0]
    summary = f"Analisi AI: {first.startTime} - {first.endTime}"
    if first.line:
        summary += f" | Linea {first.line}"
    if first.train:
        summary += f" | Treno {first.train}"
    if first.vehicle:
        summary += f" | Turno {first.vehicle}"
    if len(parsed.segments) > 1:
        summary += f" | Segmenti rilevati: {len(parsed.segments)}"
    return summary


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
    )


def map_openai_error(response: httpx.Response) -> HTTPException:
    message = response.text
    code = "openai_error"

    try:
        payload = response.json()
        error = payload.get("error", {})
        code = error.get("code") or error.get("type") or code
        message = error.get("message") or message
    except Exception:
        pass

    status_code = 502
    normalized_code = str(code).lower()
    if normalized_code == "invalid_api_key":
        status_code = 401
    elif normalized_code == "insufficient_quota":
        status_code = 402

    return api_error(status_code=status_code, code=normalized_code, message=message)


def map_gemini_error(response: httpx.Response) -> HTTPException:
    message = response.text
    code = "gemini_error"

    try:
        payload = response.json()
        error = payload.get("error", {})
        code = (error.get("status") or error.get("message") or code).lower().replace(" ", "_")
        message = error.get("message") or message
    except Exception:
        pass

    status_code = 502
    if "api_key" in code or response.status_code == 401:
        status_code = 401
        code = "invalid_api_key"
    elif "quota" in code or response.status_code == 429:
        status_code = 402
        code = "insufficient_quota"

    return api_error(status_code=status_code, code=code, message=message)
