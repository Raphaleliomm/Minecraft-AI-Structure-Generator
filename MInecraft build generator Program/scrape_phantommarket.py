#!/usr/bin/env python3
"""Lädt Builds von market.phantom-node.com als Trainingsdaten.

Geholt werden alle in der Sitemap gelisteten Builds (Titel, Beschreibung, Tags,
Autor, Versionen, Statistiken) und optional die Struktur-Dateien selbst.
Die Build-Beschreibung wird als Prompt gespeichert.

Ausgabe-Layout (direkt als --data-dir für dataset.py / train.py nutzbar):

    <out-dir>/
        builds_meta.jsonl      eine JSON-Zeile pro Build (alle Metadaten, Resume-Daten)
        prompts.json           {"<datei>.schem": "<prompt>", ...}
        <slug>__Name.schem     Struktur (konvertiert, von dataset.py gelesen)
        <slug>__Name.litematic Original-Format (optional, via --formats)
        <slug>__Name.txt       Prompt-Sidecar (von dataset.discover_examples gelesen)

Die .txt-Sidecars enthalten optional Herkunftszeilen (name:/author:/source:/url:).
Der Dataset-Parser filtert genau diese Präfixe heraus (dataset.META_PREFIXES),
mit --no-meta-lines entsteht exakt das alte Format (nur Prompt-Zeilen).

WICHTIG - RECHTLICHER HINWEIS
    market.phantom-node.com setzt in der robots.txt "Content-Signal: ai-train=no"
    und der Download-Endpoint /api/posts/*/files/*/download liegt unter
    "Disallow: /api/". Der Betreiber der Seite untersagt damit die Verwendung der
    Inhalte als KI-Trainingsdaten. Dieses Skript ändert daran nichts - die
    Verantwortung fuer Nutzung, Weitergabe und Veroeffentlichung liegt beim
    Ausfuehrenden. Bitte die Nutzungsbedingungen der Seite beachten.

Beispiele:
    python scrape_phantommarket.py --formats schem --delay 0.8
    python scrape_phantommarket.py --metadata-only --max-builds 50
    python scrape_phantommarket.py --formats litematic,schem --max-file-mb 100
    python scrape_phantommarket.py --rebuild-prompts      # prompts.json neu ableiten
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError:  # pragma: no cover - klare Meldung statt Traceback
    sys.exit("Fehlende Abhaengigkeit 'requests'. Bitte 'pip install -r requirements.txt' ausfuehren.")


# ── Konstanten ───

SITE = "https://market.phantom-node.com"
DEFAULT_SITEMAP = f"{SITE}/api/sitemap-posts.xml"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "Trainingsdaten" / "phantommarket"
USER_AGENT = "MinecraftAIStructureGenerator-DatasetScraper/1.0 (lokales Skript fuer Trainingsdaten)"

# Download-Format -> Dateiendung (Formatnamen entsprechen dem Query-Parameter der Seite)
FORMAT_SUFFIXES = {
    "litematic": ".litematic",
    "schem": ".schem",
    "nbt": ".nbt",
    "java_world": ".zip",
    "mcstructure": ".mcstructure",
    "mcpack": ".mcpack",
}
GZIP_SUFFIXES = frozenset({".schem", ".schematic", ".litematic", ".nbt"})
ZIP_SUFFIXES = frozenset({".zip", ".mcpack"})
# Formate, die dataset.discover_examples() einsammelt
DATASET_SUFFIXES = frozenset({".schem", ".schematic"})

ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
LOC_PATTERN = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


class ScrapeError(RuntimeError):
    """Fehler beim Abrufen oder Validieren einer Ressource."""


# ─── HTTP-Schicht ───

class HttpClient:
    """Rate-limitierter HTTP-Client mit Retry/Backoff für 429, 5xx und Netzfehler."""

    def __init__(
        self,
        *,
        delay: float,
        jitter: float,
        timeout: float,
        retries: int,
        user_agent: str,
    ) -> None:
        self.delay = max(0.0, delay)
        self.jitter = max(0.0, jitter)
        self.timeout = timeout
        self.retries = max(0, retries)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/octet-stream;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
            }
        )
        self.request_count = 0
        self.retry_count = 0

    def close(self) -> None:
        self.session.close()

    def _throttle(self) -> None:
        if self.delay > 0 or self.jitter > 0:
            time.sleep(self.delay + random.uniform(0.0, self.jitter))

    @staticmethod
    def _retry_after(response: "requests.Response", attempt: int) -> float:
        raw = response.headers.get("Retry-After", "").strip()
        if raw.isdigit():
            return min(120.0, float(raw))
        return min(120.0, 5.0 * (attempt + 1))

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(60.0, 2.0**attempt) + random.uniform(0.0, 1.0)

    def get(self, url: str, *, stream: bool = False, label: str = "") -> "requests.Response":
        """GET mit Rate-Limit; gibt eine 2xx/3xx-Antwort zurück, sonst ScrapeError."""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            response: "requests.Response | None" = None
            try:
                response = self.session.get(
                    url, timeout=self.timeout, allow_redirects=True, stream=stream
                )
            except requests.RequestException as exc:
                last_error = ScrapeError(f"Netzwerkfehler: {exc}")
            else:
                self.request_count += 1
                if response.status_code == 429 or response.status_code >= 500:
                    wait = self._retry_after(response, attempt)
                    last_error = ScrapeError(f"HTTP {response.status_code}")
                    response.close()
                    response = None
                    if attempt < self.retries:
                        self.retry_count += 1
                        print(
                            f"    .. HTTP {last_error} - warte {wait:.0f}s "
                            f"(Versuch {attempt + 2}/{self.retries + 1}) {label}"
                        )
                        time.sleep(wait)
                        continue
                elif 200 <= response.status_code < 400:
                    return response
                else:
                    last_error = ScrapeError(f"HTTP {response.status_code}")
                    response.close()
                    response = None

            if attempt < self.retries:
                self.retry_count += 1
                time.sleep(self._backoff(attempt))

        raise ScrapeError(f"{last_error or 'unbekannter Fehler'} ({url})")

    def get_text(self, url: str) -> str:
        response = self.get(url, label=url)
        try:
            declared = (response.encoding or "").lower()
            # requests faellt bei "text/*" ohne charset im Header auf ISO-8859-1
            # zurueck, die Seite liefert aber UTF-8 -> sonst Mojibake in Beschreibungen.
            if declared and declared not in {"iso-8859-1", "latin-1", "ascii", "us-ascii"}:
                return response.content.decode(response.encoding, errors="replace")
            try:
                return response.content.decode("utf-8")
            except UnicodeDecodeError:
                return response.content.decode(
                    response.apparent_encoding or "utf-8", errors="replace"
                )
        finally:
            response.close()


def discover_build_urls(client: HttpClient, sitemap_url: str, max_depth: int = 3) -> list[str]:
    """Läuft Sitemap-Index -> Untersitemaps ab und sammelt alle Build-URLs.

    Reihenfolge der Sitemap bleibt erhalten (neueste Builds zuerst).
    """
    seen: set[str] = set()
    build_urls: list[str] = []
    queue: list[tuple[str, int]] = [(sitemap_url, 0)]

    while queue:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        xml = client.get_text(url)
        locations = LOC_PATTERN.findall(xml)
        for location in locations:
            if location.endswith(".xml"):
                queue.append((location, depth + 1))
            else:
                build_urls.append(location)

    unique = list(dict.fromkeys(build_urls))
    print(f"Sitemap: {len(unique)} Build-URLs gefunden ({len(seen)} Sitemap-Datei(en))")
    return unique


def slug_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


# ─── Parsing der Build-Seite ───

def extract_initial_post(html_text: str) -> dict[str, Any] | None:
    """Holt den 'initialPost'-Block aus den Astro-Island-Props der Build-Seite."""
    for raw_props in re.findall(r'props="([^"]*)"', html_text):
        if "initialPost" not in raw_props:
            continue
        try:
            data = json.loads(html_module.unescape(raw_props))
        except json.JSONDecodeError:
            continue
        unboxed = unbox_devalue(data)
        if isinstance(unboxed, dict) and isinstance(unboxed.get("initialPost"), dict):
            return unboxed["initialPost"]
    return None


def extract_meta_content(html_text: str, name: str) -> str:
    match = re.search(
        rf'<meta[^>]+(?:name|property)="{re.escape(name)}"[^>]+content="([^"]*)"',
        html_text,
    )
    return html_module.unescape(match.group(1)) if match else ""


def extract_rendered_description(html_text: str) -> str:
    """Fällt auf den serverseitig gerenderten Beschreibungstext zurück."""
    match = re.search(r'<div class="post-content markdown-body">(.*?)</div>', html_text, re.S)
    if not match:
        return ""
    return markdown_to_text(match.group(1))


def resolve_description(html_text: str, post: dict[str, Any]) -> str:
    """Beschreibung aus den Island-Props, sonst aus dem HTML, sonst aus og:description.

    Die Build-Beschreibung ist der Prompt - deshalb lieber eine ungenaue Variante
    als gar keine (dataset.discover_examples braucht einen Prompt pro Struktur).
    """
    description = str(post.get("description") or "").strip()
    if description:
        return description
    rendered = extract_rendered_description(html_text)
    if rendered:
        return rendered
    return extract_meta_content(html_text, "og:description")

# ─── Text- und Wert-Helfer (von Parser und Downloader genutzt) ───

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_filename(stem: str, max_length: int = 90) -> str:
    """Macht aus einem Build-Dateinamen einen Windows-tauglichen Namensteil."""
    name = ILLEGAL_FILENAME_CHARS.sub(" ", str(stem))
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    if len(name) > max_length:
        name = name[:max_length].strip()
    return name or "structure"


def unbox_devalue(value: Any) -> Any:
    """Entpackt Astros devalue-Serialisierung ([tag, wert]) zu normalen Werten.

    Die Server-Islands liefern Metadaten als Wrapper-Listen, z. B.
    {"title": [0, "Ash Castle"], "tags": [1, [[0, {"name": [0, "castle"]}]]]}.
    """
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], int)
        and 0 <= value[0] <= 6
    ):
        tag, payload = value
        if tag in (1, 2):  # Array / Set
            return [unbox_devalue(item) for item in payload]
        if tag == 4:  # Map
            return {unbox_devalue(key): unbox_devalue(item) for key, item in payload}
        return unbox_devalue(payload)
    if isinstance(value, dict):
        return {key: unbox_devalue(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unbox_devalue(item) for item in value]
    return value


_MD_FENCE = re.compile(r"```.*?```", re.S)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_LIST = re.compile(r"(?m)^\s*(?:#{1,6}|>|[-+*]|\d+[.)])\s*")
_MD_MARKS = re.compile(r"[*_`~]+")
_HTML_TAG = re.compile(r"<[^>]+>")


def markdown_to_text(text: str) -> str:
    """Reduziert Markdown/HTML auf reinen Fliesstext."""
    text = _MD_FENCE.sub(" ", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = html_module.unescape(text)
    text = _MD_LIST.sub("", text)
    text = text.replace("|", " ")
    text = _MD_MARKS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clamp_text(text: str, max_chars: int) -> str:
    """Kürzt an einer Wortgrenze, ohne mitten im Wort abzuschneiden."""
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].strip(" ,;:-")
    return cut or text[:max_chars].strip()


def build_prompt(
    description: str, title: str, tags: Iterable[str], mode: str, max_chars: int
) -> tuple[str, bool]:
    """Baut den Prompt aus der Build-Beschreibung. Returns (prompt, war_fallback)."""
    desc = markdown_to_text(description or "")
    title = (title or "").strip()
    tag_text = ", ".join(list(tags)[:6])

    if mode == "title":
        primary, fallbacks = title, (desc, tag_text)
    elif mode == "title+description":
        primary = f"{title}: {desc}" if title and desc else (title or desc)
        fallbacks = (title, desc, tag_text)
    else:  # "description" - die Beschreibung ist der Prompt
        primary, fallbacks = desc, (title, tag_text)

    prompt = clamp_text(primary, max_chars)
    if prompt:
        return prompt, False
    for fallback in fallbacks:
        prompt = clamp_text(fallback, max_chars)
        if prompt:
            return prompt, True
    return "", True
# ─── Build-Datensatz aus der Seite bauen ───

def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_build_page(
    html_text: str, source_url: str, *, prompt_mode: str, max_prompt_chars: int
) -> dict[str, Any]:
    """Baut den Metadaten-Datensatz (inkl. Prompt) aus dem HTML einer Build-Seite."""
    slug = slug_from_url(source_url)
    post = extract_initial_post(html_text)
    if post is None:
        raise ScrapeError("keine initialPost-Daten im HTML gefunden")

    title = str(post.get("title") or "").strip()
    description = resolve_description(html_text, post)

    raw_tags = [tag for tag in (post.get("tags") or []) if isinstance(tag, dict)]
    tag_names = [str(tag.get("name")).strip() for tag in raw_tags if tag.get("name")]
    first_tag = raw_tags[0] if raw_tags else {}

    author = post.get("author") or {}
    author_name = ""
    if isinstance(author, dict):
        author_name = str(author.get("username") or author.get("id") or "").strip()

    files: list[dict[str, Any]] = []
    for entry in post.get("files") or []:
        if not isinstance(entry, dict) or not entry.get("fileHash"):
            continue
        files.append(
            {
                "filename": str(entry.get("filename") or ""),
                "file_type": str(entry.get("fileType") or "").strip().lower(),
                "file_hash": str(entry.get("fileHash")),
                "file_size": _to_int(entry.get("fileSize")),
                "download_count": _to_int(entry.get("downloadCount")),
            }
        )

    mc_versions = [
        str(version.get("version"))
        for version in (post.get("versions") or [])
        if isinstance(version, dict) and version.get("version")
    ]

    prompt, used_fallback = build_prompt(description, title, tag_names, prompt_mode, max_prompt_chars)

    return {
        "slug": slug,
        "url": source_url,
        "source": "phantommarket",
        "title": title,
        "description": description,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_mode": prompt_mode,
        "prompt_fallback": used_fallback,
        "tags": tag_names,
        "category": str(first_tag.get("category") or "").strip(),
        "type": str(first_tag.get("type") or "").strip(),
        "author": author_name,
        "created_at": str(post.get("createdAt") or ""),
        "updated_at": str(post.get("updatedAt") or ""),
        "mc_versions": mc_versions[:8],
        "has_mods": bool(post.get("hasMods")),
        "mod_loader": str(post.get("modLoader") or ""),
        "is_original_work": bool(post.get("isOriginalWork")),
        "stats": {
            "views": _to_int(post.get("viewCount")),
            "likes": _to_int(post.get("likeCount")),
            "downloads": _to_int(post.get("downloadCount")),
            "saves": _to_int(post.get("savesCount")),
            "comments": _to_int(post.get("commentCount")),
        },
        "files": files,
        "scraped_at": now_iso(),
    }


# ─── Download und Validierung ───

def target_file_name(slug: str, filename: str, suffix: str) -> str:
    """Eindeutiger Dateiname: <slug>__<originaler Name><endung>.

    Der Slug-Prefix verhindert Kollisionen (viele Builds heissen 'Medieval House').
    """
    stem = Path(filename).name
    if stem.lower().endswith(suffix.lower()):
        stem = stem[: -len(suffix)]
    elif "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return f"{slug}__{sanitize_filename(stem)}{suffix}"


def looks_like_structure_file(head: bytes, suffix: str) -> bool:
    """Plausibilitätsprüfung der ersten Bytes (statt Blind-Download von Fehlerseiten)."""
    if not head:
        return False
    lowered = head[:64].lower()
    if head[:1] in (b"{", b"<") or b"<!doctype" in lowered or b"<html" in lowered:
        return False
    if suffix in GZIP_SUFFIXES:
        # .schem/.litematic/.nbt sind gzip-komprimiertes oder rohes NBT
        return head[:2] == b"\x1f\x8b" or head[:1] == b"\x0a"
    if suffix in ZIP_SUFFIXES:
        return head[:4] == b"PK\x03\x04"
    return len(head) >= 8


def download_file(
    client: HttpClient,
    *,
    slug: str,
    file_info: dict[str, Any],
    format_name: str,
    target: Path,
    max_file_mb: float,
    redownload: bool,
) -> dict[str, Any]:
    """Lädt eine Struktur-Datei (ggf. konvertiert) herunter und validiert sie."""
    suffix = FORMAT_SUFFIXES[format_name]
    source_type = file_info["file_type"]
    size_hint = file_info["file_size"]

    if size_hint and max_file_mb > 0 and size_hint > max_file_mb * 1024 * 1024:
        return {
            "format": format_name,
            "file": target.name,
            "status": "skipped_too_large",
            "bytes": 0,
            "source_bytes": size_hint,
        }

    if target.exists() and not redownload:
        with target.open("rb") as handle:
            head = handle.read(16)
        if looks_like_structure_file(head, suffix):
            return {
                "format": format_name,
                "file": target.name,
                "status": "existing",
                "bytes": target.stat().st_size,
            }

    # Original-Format ohne Konvertierungs-Parameter laden, sonst ?format=<name>
    query = "" if format_name == source_type else f"?format={format_name}"
    url = f"{SITE}/api/posts/{slug}/files/{file_info['file_hash']}/download{query}"

    response = client.get(url, stream=True, label=f"{slug} [{format_name}]")
    content_type = response.headers.get("content-type", "?")
    temp_path = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    written = 0
    head = b""
    try:
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(65536):
                if not chunk:
                    continue
                if not head:
                    head = chunk[:16]
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    finally:
        response.close()

    if not looks_like_structure_file(head, suffix):
        temp_path.unlink(missing_ok=True)
        raise ScrapeError(
            f"{suffix}-Download ungueltig (content-type={content_type}, {written} Bytes) - "
            "Format evtl. noch nicht konvertiert"
        )

    expected = _to_int(response.headers.get("content-length"))
    if expected and written != expected:
        temp_path.unlink(missing_ok=True)
        raise ScrapeError(f"unvollstaendiger Download ({written}/{expected} Bytes)")

    temp_path.replace(target)

    file_digest = digest.hexdigest()
    verified: bool | None = None
    if format_name == source_type:
        verified = file_digest == file_info["file_hash"]

    return {
        "format": format_name,
        "file": target.name,
        "status": "downloaded",
        "bytes": written,
        "sha256": file_digest,
        "sha256_matches_source": verified,
    }


# ─── Schreiben: Prompt-Sidecar, builds_meta.jsonl, prompts.json ───

META_FILENAME = "builds_meta.jsonl"
PROMPTS_FILENAME = "prompts.json"


def write_prompt_sidecar(
    path: Path, prompt: str, record: dict[str, Any], with_meta_lines: bool
) -> None:
    """Schreibt <struktur>.txt mit dem Prompt (Format von dataset.discover_examples).

    Die Praefixe name:/author:/source:/url: filtert der Dataset-Parser als
    Metadaten heraus (dataset.META_PREFIXES), der Rest gilt als Prompt.
    """
    if not prompt:
        return
    lines: list[str] = []
    if with_meta_lines:
        if record.get("title"):
            lines.append(f"name: {record['title']}")
        if record.get("author"):
            lines.append(f"author: {record['author']}")
        lines.append("source: phantommarket")
        lines.append(f"url: {record['url']}")
        lines.append("---")
    lines.append(prompt)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_meta(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Liest builds_meta.jsonl für den Resume (letzte Zeile je Slug gewinnt)."""
    meta_path = out_dir / META_FILENAME
    records: dict[str, dict[str, Any]] = {}
    if not meta_path.exists():
        return records
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = record.get("slug")
            if slug:
                records[slug] = record
    return records


def append_meta(out_dir: Path, record: dict[str, Any]) -> None:
    """Hängt einen Datensatz an builds_meta.jsonl an (crash-sicher für Resume)."""
    with (out_dir / META_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def collect_prompt_map(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Baut {dateiname: prompt} aus allen Datensätzen (nur dataset-Formate)."""
    prompt_map: dict[str, str] = {}
    for record in records:
        prompt = record.get("prompt")
        if not prompt:
            continue
        for name in record.get("prompt_files") or []:
            prompt_map[name] = prompt
    return prompt_map


def write_prompts_json(out_dir: Path, records: Iterable[dict[str, Any]]) -> int:
    """Schreibt prompts.json atomar und gibt die Anzahl der Einträge zurück."""
    prompt_map = collect_prompt_map(records)
    path = out_dir / PROMPTS_FILENAME
    temp_path = path.with_name(path.name + ".tmp")
    payload = json.dumps(prompt_map, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.write_text(payload + "\n", encoding="utf-8")
    temp_path.replace(path)
    return len(prompt_map)


def record_is_complete(
    record: dict[str, Any], out_dir: Path, *, metadata_only: bool
) -> bool:
    """Prüft, ob ein Build bereits vollständig geholt wurde (Resume-Entscheidung)."""
    if not record.get("ok"):
        return False
    if metadata_only or not record.get("files"):
        return True
    downloads = record.get("downloads") or []
    if not downloads:
        return False
    for entry in downloads:
        if entry.get("status") not in ("downloaded", "existing"):
            return False
        relative = entry.get("path")
        if entry.get("status") in ("downloaded", "existing"):
            if not relative or not (out_dir / relative).exists():
                return False
    return True


# ─── Ein Build komplett holen ───

def scrape_build(
    client: HttpClient,
    *,
    build_url: str,
    out_dir: Path,
    formats: list[str],
    args: argparse.Namespace,
    hash_index: dict[str, str],
) -> dict[str, Any]:
    """Holt Seite und Dateien eines Builds und liefert den Metadaten-Datensatz."""
    html_text = client.get_text(build_url)
    record = parse_build_page(
        html_text,
        build_url,
        prompt_mode=args.prompt_mode,
        max_prompt_chars=args.max_prompt_chars,
    )
    record["formats"] = list(formats)

    target_dir = out_dir
    if args.split_by_category:
        target_dir = out_dir / sanitize_filename(record.get("category") or "uncategorized")
    target_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[dict[str, Any]] = []
    errors: list[str] = []
    prompt_files: list[str] = []
    sidecar_names: set[str] = set()

    for file_info in record["files"]:
        for format_name in formats:
            suffix = FORMAT_SUFFIXES[format_name]
            name = target_file_name(record["slug"], file_info["filename"], suffix)
            target = target_dir / name
            relative = str(target.relative_to(out_dir)).replace("\\", "/")
            sidecar_names.add(Path(name).with_suffix(".txt").name)

            if suffix in DATASET_SUFFIXES and name not in prompt_files:
                prompt_files.append(name)

            # Duplikat-Erkennung anhand des Server-Hashes (nur im Original-Format)
            if args.dedupe_by_hash and format_name == file_info["file_type"]:
                owner = hash_index.get(file_info["file_hash"])
                if owner and owner != record["slug"]:
                    downloads.append(
                        {
                            "format": format_name,
                            "file": name,
                            "path": relative,
                            "status": "duplicate",
                            "duplicate_of": owner,
                            "bytes": 0,
                        }
                    )
                    continue
                hash_index.setdefault(file_info["file_hash"], record["slug"])

            if args.metadata_only:
                continue

            try:
                entry = download_file(
                    client,
                    slug=record["slug"],
                    file_info=file_info,
                    format_name=format_name,
                    target=target,
                    max_file_mb=args.max_file_mb,
                    redownload=args.redownload,
                )
            except (ScrapeError, OSError) as exc:
                errors.append(f"{name}: {exc}")
                entry = {
                    "format": format_name,
                    "file": name,
                    "status": "error",
                    "error": str(exc),
                    "bytes": 0,
                }
            entry["path"] = relative
            downloads.append(entry)

    # Prompt-Sidecars je Basisname (ein .txt fuer .schem + .litematic desselben Builds)
    sidecar_paths: list[str] = []
    for sidecar_name in sorted(sidecar_names):
        sidecar_path = target_dir / sidecar_name
        write_prompt_sidecar(
            sidecar_path, record.get("prompt", ""), record, args.meta_lines
        )
        sidecar_paths.append(str(sidecar_path.relative_to(out_dir)).replace("\\", "/"))

    record["downloads"] = downloads
    record["prompt_files"] = prompt_files
    record["sidecar_files"] = sidecar_paths
    record["errors"] = errors
    record["ok"] = not errors
    record["scraped_at"] = now_iso()
    return record


# ─── CLI ───

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laedt PhantomMarket-Builds als Trainingsdaten (Build-Beschreibung = Prompt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Zielordner fuer Daten und Prompt-Dateien")
    parser.add_argument("--sitemap", default=DEFAULT_SITEMAP, help="Sitemap mit allen Build-URLs")
    parser.add_argument("--formats", default="schem", help=f"Komma-Liste aus: {', '.join(FORMAT_SUFFIXES)}")
    parser.add_argument("--max-builds", type=int, default=0, help="Nur die ersten N Builds der Sitemap laden (0 = alle)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Lauf beenden, nachdem N Builds verarbeitet wurden (0 = kein Limit, Resume-freundlich)",
    )
    parser.add_argument("--max-file-mb", type=float, default=250.0, help="Groessere Dateien ueberspringen (0 = kein Limit)")
    parser.add_argument("--delay", type=float, default=0.8, help="Pause zwischen zwei Requests in Sekunden")
    parser.add_argument("--jitter", type=float, default=0.4, help="Zufaelliger Zusatz zur Pause in Sekunden")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP-Timeout in Sekunden")
    parser.add_argument("--retries", type=int, default=4, help="Wiederholungen bei 429/5xx/Netzfehlern")
    parser.add_argument("--user-agent", default=USER_AGENT, help="User-Agent der Anfragen")
    parser.add_argument(
        "--prompt-mode",
        choices=("description", "title", "title+description"),
        default="description",
        help="Woraus der Prompt gebaut wird",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=0,
        help="Prompt-Laenge kappen (0 = ungekuerzt, Beschreibung unveraendert uebernehmen)",
    )
    parser.add_argument("--order", choices=("newest", "oldest", "shuffle"), default="newest", help="Reihenfolge der Builds")
    parser.add_argument("--metadata-only", action="store_true", help="Nur Beschreibungen/Metadaten, keine Struktur-Dateien")
    parser.add_argument("--refresh-meta", action="store_true", help="Metadaten vorhandener Builds erneut laden")
    parser.add_argument("--redownload", action="store_true", help="Vorhandene Struktur-Dateien erneut laden")
    parser.add_argument("--dedupe-by-hash", action="store_true", help="Dateien mit bereits geladenem Server-Hash ueberspringen")
    parser.add_argument("--split-by-category", action="store_true", help="Dateien in Unterordner je Kategorie legen")
    parser.add_argument(
        "--meta-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="name:/author:/source:/url:-Zeilen in die .txt-Sidecars schreiben (--no-meta-lines = Altformat)",
    )
    parser.add_argument("--rebuild-prompts", action="store_true", help="prompts.json neu aus builds_meta.jsonl ableiten und beenden")
    return parser.parse_args(argv)


def describe_activity(record: dict[str, Any]) -> str:
    """Kurze Statuszeile fuer die Konsole."""
    downloads = record.get("downloads") or []
    fresh = [entry for entry in downloads if entry.get("status") == "downloaded"]
    existing = [entry for entry in downloads if entry.get("status") == "existing"]
    duplicates = [entry for entry in downloads if entry.get("status") == "duplicate"]
    failed = [entry for entry in downloads if entry.get("status") == "error"]
    if not downloads:
        return "metadaten"
    parts: list[str] = []
    if fresh:
        total_mb = sum(_to_int(entry.get("bytes")) for entry in fresh) / (1024 * 1024)
        parts.append(f"{len(fresh)} Datei(en) {total_mb:.1f} MB")
    if existing:
        parts.append(f"{len(existing)} vorhanden")
    if duplicates:
        parts.append(f"{len(duplicates)} Duplikat")
    if failed:
        parts.append(f"{len(failed)} Fehler")
    return ", ".join(parts) or "nichts zu tun"


# ─── Ablauf ───

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    if args.rebuild_prompts:
        records = load_meta(out_dir)
        entries = write_prompts_json(out_dir, records.values())
        print(f"prompts.json neu geschrieben: {entries} Prompts aus {len(records)} Builds")
        print(f"Datei: {out_dir / PROMPTS_FILENAME}")
        return 0

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    if not formats:
        print("Fehler: --formats darf nicht leer sein.", file=sys.stderr)
        return 2
    unknown = [item for item in formats if item not in FORMAT_SUFFIXES]
    if unknown:
        print(
            f"Fehler: unbekannte Formate {unknown}. Erlaubt: {', '.join(FORMAT_SUFFIXES)}",
            file=sys.stderr,
        )
        return 2

    print("PhantomMarket-Scraper (Build-Beschreibung = Prompt)")
    print(f"Ziel      : {out_dir}")
    print(f"Formate   : {', '.join(formats)}{' (nur Metadaten)' if args.metadata_only else ''}")
    print(f"Rate-Limit: {args.delay}s + {args.jitter}s Jitter | Retries: {args.retries}")

    client = HttpClient(
        delay=args.delay,
        jitter=args.jitter,
        timeout=args.timeout,
        retries=args.retries,
        user_agent=args.user_agent,
    )
    counters = {"new": 0, "skipped": 0, "failed": 0, "files": 0, "bytes": 0}
    records = load_meta(out_dir)
    interrupted = False
    processed = 0

    try:
        build_urls = discover_build_urls(client, args.sitemap)
        if args.order == "oldest":
            build_urls.reverse()
        elif args.order == "shuffle":
            random.shuffle(build_urls)
        if args.max_builds > 0:
            build_urls = build_urls[: args.max_builds]

        hash_index: dict[str, str] = {}
        for record in records.values():
            if not record.get("ok"):
                continue
            for file_info in record.get("files") or []:
                if file_info.get("file_hash"):
                    hash_index[file_info["file_hash"]] = record["slug"]

        total = len(build_urls)
        print(f"Zu verarbeiten: {total} Builds | bereits gespeichert: {len(records)}")
        print("-" * 72)

        for position, build_url in enumerate(build_urls, start=1):
            slug = slug_from_url(build_url)
            previous = None if args.refresh_meta else records.get(slug)
            if previous and record_is_complete(
                previous, out_dir, metadata_only=args.metadata_only
            ):
                counters["skipped"] += 1
                print(f"[{position}/{total}] {slug} :: schon vorhanden")
                continue

            print(f"[{position}/{total}] {slug} :: ", end="", flush=True)
            try:
                record = scrape_build(
                    client,
                    build_url=build_url,
                    out_dir=out_dir,
                    formats=formats,
                    args=args,
                    hash_index=hash_index,
                )
            except (ScrapeError, OSError, ValueError) as exc:
                record = {
                    "slug": slug,
                    "url": build_url,
                    "source": "phantommarket",
                    "ok": False,
                    "errors": [str(exc)],
                    "prompt": "",
                    "files": [],
                    "downloads": [],
                    "scraped_at": now_iso(),
                }
                counters["failed"] += 1
                print(f"FEHLER: {exc}")
            else:
                fresh = [
                    entry
                    for entry in record["downloads"]
                    if entry.get("status") == "downloaded"
                ]
                counters["files"] += len(fresh)
                counters["bytes"] += sum(_to_int(entry.get("bytes")) for entry in fresh)
                counters["new" if record["ok"] else "failed"] += 1
                print(f"{'ok' if record['ok'] else 'mit Fehlern'} :: {describe_activity(record)}")

            records[slug] = record
            append_meta(out_dir, record)

            processed += 1
            if args.limit and processed >= args.limit:
                print(f"Limit erreicht: {processed} Build(s) verarbeitet - Lauf beendet.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nAbgebrochen (Strg+C).")
    finally:
        client.close()

    prompt_count = write_prompts_json(out_dir, records.values())

    print("-" * 72)
    print("Ergebnis")
    print(f"  neue/aktualisierte Builds : {counters['new']}")
    print(f"  uebersprungen             : {counters['skipped']}")
    print(f"  Builds mit Fehlern        : {counters['failed']}")
    print(f"  Struktur-Dateien geladen  : {counters['files']} ({counters['bytes'] / (1024 * 1024):.1f} MB)")
    print(f"  Prompts in prompts.json   : {prompt_count}")
    print(f"  HTTP-Requests             : {client.request_count} (Retries: {client.retry_count})")
    print(f"  Ausgabe                   : {out_dir}")

    if interrupted:
        print("\nResume: denselben Befehl erneut starten - vorhandene Builds werden uebersprungen.")
    elif counters["failed"]:
        print("\nFehlgeschlagene Builds stehen mit ok=false in builds_meta.jsonl und werden erneut versucht.")

    if not args.metadata_only:
        print(
            "\nAls Trainingsdaten nutzen:"
            f'\n  python train.py --data-dir "{out_dir}" --model-size 16 --epochs 10'
        )
    print("\nHinweis: Die Seite untersagt in ihrer robots.txt die Nutzung als KI-Trainingsdaten (ai-train=no).")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
