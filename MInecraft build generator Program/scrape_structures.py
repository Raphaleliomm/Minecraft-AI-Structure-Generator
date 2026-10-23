#!/usr/bin/env python3
"""Sammelt Minecraft-Strukturen aus mehreren freien Quellen als Trainingsdaten.

Unterstuetzt aktuell:

* ``abfielder``   - abfielder.com (groesstes freies Archiv, .litematic)
* ``huggingface`` - Hugging-Face-Datensatz ``Hack337/Minecraft-Schematics``
                    (MIT: rohe .schem/.litematic + JSON-Blueprints)

Beide Quellen liefern ein Layout, das ``dataset.py`` direkt lesen kann::

    <out-root>/<quelle>/
        structures_meta.jsonl      eine JSON-Zeile pro Struktur (Resume + Herkunft)
        prompts.json               {"<datei>.schem": ["<name>", "<prompt>"]}
        <quelle>-<id>__<name>.schem
        <quelle>-<id>__<name>.txt  Prompt-Sidecar (Name / '---' / Beschreibung)

Litematica-Dateien werden lokal nach Sponge-.schem konvertiert
(``app/litematic.py``), damit keine externen Konverter noetig sind.

Rechtliches / Fairness
    Die grossen Portale (Planet Minecraft, minecraft-schematics.com,
    MineSchematic, Shulkr, Bloxelizer) verlangen einen Login, sperren
    KI-Crawler in der robots.txt oder liefern nur ueber gesperrte API-Pfade.
    Dieses Skript respektiert ``robots.txt`` standardmaessig, haelt
    Rate-Limits ein und laedt nur frei zugaengliche Dateien. Die Rechte an
    den Builds liegen bei den jeweiligen Creator:innen; Herkunft und Lizenz
    stehen in ``structures_meta.jsonl``.

Beispiele:
    python scrape_structures.py --list-sources
    python scrape_structures.py --source abfielder --max-items 25 --delay 2.0
    python scrape_structures.py --source huggingface
    python scrape_structures.py --source all --metadata-only
    python scrape_structures.py --source huggingface --no-blueprints
    python scrape_structures.py --source abfielder --rebuild-prompts
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html as html_module
import io
import json
import random
import re
import shutil
import sys
import time
import unicodedata
import urllib.parse
import urllib.robotparser
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("Fehlende Abhaengigkeit 'requests'. Bitte 'pip install -r requirements.txt' ausfuehren.")

from app.litematic import (
    AIR,
    StructureConversionError,
    StructureTooLarge,
    blueprint_to_schem,
    detect_structure_format,
    litematic_to_schem,
    read_structure_size,
)


# ── Konstanten ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = PROJECT_ROOT.parent / "Trainingsdaten"
META_FILENAME = "structures_meta.jsonl"
PROMPTS_FILENAME = "prompts.json"
PROMPT_SEPARATOR = "---"
DATASET_SUFFIXES = frozenset({".schem", ".schematic"})
USER_AGENT = (
    "MinecraftAIStructureGenerator-DatasetScraper/1.1 "
    "(lokales Skript fuer Trainingsdaten; respektiert robots.txt)"
)
ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
DASH_RUN_PATTERN = re.compile(r"-{3,}")
LOC_PATTERN = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
HTML_TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


class ScrapeError(RuntimeError):
    """Fehler beim Abrufen, Parsen oder Validieren einer Ressource."""


class RobotsDisallowed(ScrapeError):
    """Die robots.txt der Seite verbietet den Abruf."""


# ── Text-/Datei-Helfer ─────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def html_to_text(fragment: str) -> str:
    """HTML-Fragment -> reiner Fliesstext."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", fragment)
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"(?i)</p\s*>", " ", text)
    text = HTML_TAG.sub(" ", text)
    text = html_module.unescape(text)
    return WHITESPACE.sub(" ", text).strip()


def clamp_text(text: str, max_chars: int) -> str:
    text = WHITESPACE.sub(" ", str(text or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].strip(" ,;:-")
    return cut or text[:max_chars].strip()


def sanitize_filename(stem: str, max_length: int = 90) -> str:
    name = ILLEGAL_FILENAME_CHARS.sub(" ", str(stem))
    name = WHITESPACE.sub(" ", name).strip().strip(".")
    if len(name) > max_length:
        name = name[:max_length].strip()
    return name or "structure"


def slugify(text: str, max_length: int = 60) -> str:
    """ASCII-Slug fuer Dateinamen (Titel bleiben im Sidecar unveraendert)."""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or "structure"


def clean_sidecar_name(name: str) -> str:
    return WHITESPACE.sub(" ", DASH_RUN_PATTERN.sub(" ", str(name))).strip()


def sidecar_prompts(title: str, prompt: str, *, with_name: bool = True) -> list[str]:
    """Prompt-Bloecke: erst der Name, dann die Beschreibung (ohne Dubletten)."""
    title = clean_sidecar_name(title)
    prompt = str(prompt or "").strip()
    blocks: list[str] = []
    if with_name and title and title != prompt:
        blocks.append(title)
    if prompt and prompt not in blocks:
        blocks.append(prompt)
    return blocks


def write_prompt_sidecar(path: Path, prompts: Iterable[str]) -> None:
    blocks = [str(item).strip() for item in prompts if str(item).strip()]
    if not blocks:
        return
    path.write_text(f"\n{PROMPT_SEPARATOR}\n".join(blocks) + "\n", encoding="utf-8")


def build_stem(source: str, identifier: str, title: str) -> str:
    ident = slugify(identifier, 28)
    if len(ident) >= 20 and re.fullmatch(r"[a-z0-9-]+", ident or ""):
        ident = ident[-12:]
    title_slug = slugify(title, 60)
    return sanitize_filename(f"{source}-{ident}__{title_slug}", 110)


def relative_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def filename_from_disposition(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", value, re.I)
    if match:
        return urllib.parse.unquote(match.group(1).strip().strip('"'))
    match = re.search(r'filename\s*=\s*"?([^";]+)"?', value, re.I)
    return match.group(1).strip() if match else ""


# ── HTTP ───────────────────────────────────────────────────────────────────

@dataclass
class Download:
    path: Path
    filename: str
    content_type: str
    size: int
    sha256: str


class HttpClient:
    """Rate-limitierter Client mit Retry/Backoff und Cloudflare-Erkennung."""

    def __init__(
        self,
        *,
        delay: float,
        jitter: float,
        timeout: float,
        retries: int,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.delay = max(0.0, delay)
        self.jitter = max(0.0, jitter)
        self.timeout = timeout
        self.retries = max(0, retries)
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/json,"
                          "application/octet-stream;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
            }
        )
        self._last_request: dict[str, float] = {}
        self.request_count = 0
        self.retry_count = 0
        self.bytes_downloaded = 0

    def close(self) -> None:
        self.session.close()

    def _throttle(self, url: str) -> None:
        host = urllib.parse.urlparse(url).netloc
        wait = self.delay + random.uniform(0.0, max(0.0, self.jitter))
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < wait:
            time.sleep(wait - elapsed)
        self._last_request[host] = time.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(90.0, 4.0 * (2**attempt)) + random.uniform(0.0, 1.0)

    @staticmethod
    def _retry_after(response: "requests.Response", attempt: int) -> float:
        raw = (response.headers.get("Retry-After") or "").strip()
        if raw.isdigit():
            return min(180.0, float(raw))
        return min(180.0, 10.0 * (attempt + 1))

    @staticmethod
    def _is_cloudflare(response: "requests.Response", body: str = "") -> bool:
        server = (response.headers.get("Server") or "").lower()
        if "cloudflare" not in server and "cf-ray" not in response.headers:
            return False
        lowered = body[:4000].lower()
        return any(
            marker in lowered
            for marker in ("just a moment", "cf-chl", "challenge-platform", "attention required")
        ) or response.status_code in (403, 429, 503)

    def get(
        self,
        url: str,
        *,
        stream: bool = False,
        headers: dict[str, str] | None = None,
        allow_404: bool = False,
    ) -> "requests.Response":
        last_error = ""
        for attempt in range(self.retries + 1):
            wait = 0.0
            self._throttle(url)
            response = None
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    stream=stream,
                    headers=headers,
                )
            except requests.RequestException as exc:
                last_error = f"Netzwerkfehler: {exc}"
            else:
                self.request_count += 1
                status = response.status_code
                if 200 <= status < 400:
                    return response
                if allow_404 and status in (404, 410):
                    return response
                if status == 429 or 500 <= status < 600:
                    wait = self._retry_after(response, attempt)
                    last_error = f"HTTP {status}"
                    response.close()
                    response = None
                elif status in (401, 403, 503):
                    body = ""
                    if not stream:
                        try:
                            body = response.text
                        except Exception:
                            body = ""
                    if self._is_cloudflare(response, body):
                        wait = min(240.0, 30.0 * (attempt + 1))
                        last_error = "Cloudflare-Challenge"
                        response.close()
                        response = None
                    else:
                        body = body or "(kein Body gelesen)"
                        response.close()
                        raise ScrapeError(f"HTTP {status} ({url})")
                else:
                    response.close()
                    raise ScrapeError(f"HTTP {status} ({url})")

            if attempt < self.retries:
                self.retry_count += 1
                wait = locals().get("wait") or self._backoff(attempt)
                print(
                    f"    .. {last_error} - warte {wait:.0f}s "
                    f"(Versuch {attempt + 2}/{self.retries + 1})"
                )
                time.sleep(wait)
            else:
                break
        raise ScrapeError(f"{last_error or 'unbekannter Fehler'} ({url})")

    def _decode_response(self, response: "requests.Response") -> str:
        """UTF-8 zuerst; nur bei echten Dekodierfehlern auf Charset raten.

        Viele Seiten liefern UTF-8 ohne charset im Header. ``requests`` faellt
        dann auf ISO-8859-1 zurueck und macht aus Emojis Mojibake. Einzelne
        fehlerhafte Bytes sollen nicht die gesamte Seite auf CP1252 umstellen,
        daher zuerst ``utf-8`` mit ``errors="replace"``.
        """
        content = response.content
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")
            # Bei sehr vielen Ersatzzeichen ist die Seite wirklich anders kodiert.
            if text.count("\ufffd") > max(8, len(text) // 200):
                fallback = response.apparent_encoding or "utf-8"
                if fallback.lower() not in {"utf-8", "ascii"}:
                    return content.decode(fallback, errors="replace")
            return text

    def get_text(self, url: str, **kwargs: Any) -> str:
        response = self.get(url, **kwargs)
        try:
            return self._decode_response(response)
        finally:
            response.close()

    def get_text_with_url(self, url: str) -> tuple[str, str]:
        response = self.get(url)
        try:
            return self._decode_response(response), response.url
        finally:
            response.close()

    def get_json(self, url: str) -> Any:
        text = self.get_text(url)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScrapeError(f"Ungueltiges JSON von {url}: {exc}") from exc

    def download(
        self,
        url: str,
        target: Path,
        *,
        max_bytes: int = 0,
        headers: dict[str, str] | None = None,
    ) -> Download:
        response = self.get(url, stream=True, headers=headers)
        content_type = response.headers.get("content-type", "")
        disposition = response.headers.get("content-disposition", "")
        filename = filename_from_disposition(disposition)
        expected = 0
        try:
            expected = int(response.headers.get("content-length") or 0)
        except ValueError:
            expected = 0
        if max_bytes and expected > max_bytes:
            response.close()
            raise ScrapeError(f"Datei zu gross ({expected / 1024 / 1024:.1f} MB)")

        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        written = 0
        head = b""
        try:
            with temp.open("wb") as handle:
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue
                    if not head:
                        head = chunk[:16]
                    written += len(chunk)
                    if max_bytes and written > max_bytes:
                        raise ScrapeError(f"Datei zu gross (>{max_bytes / 1024 / 1024:.1f} MB)")
                    handle.write(chunk)
                    digest.update(chunk)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        finally:
            response.close()

        if expected and written != expected:
            temp.unlink(missing_ok=True)
            raise ScrapeError(f"unvollstaendiger Download ({written}/{expected} Bytes)")
        if head.lstrip()[:1] in (b"<", b"{") or b"<!doctype" in head.lower():
            temp.unlink(missing_ok=True)
            raise ScrapeError("Download ist eine HTML/JSON-Seite statt einer Strukturdatei")
        if not filename:
            filename = Path(urllib.parse.urlparse(url).path).name or target.name
        temp.replace(target)
        self.bytes_downloaded += written
        return Download(target, filename, content_type, written, digest.hexdigest())


# ── robots.txt ─────────────────────────────────────────────────────────────

class RobotsPolicy:
    """Cache fuer robots.txt; 404/410 gilt als "alles erlaubt"."""

    def __init__(self, client: HttpClient, user_agent: str, enabled: bool = True) -> None:
        self.client = client
        self.user_agent = user_agent
        self.enabled = enabled
        self._cache: dict[str, tuple[bool, urllib.robotparser.RobotFileParser | None]] = {}
        self._warned: set[str] = set()

    def allowed(self, url: str) -> bool:
        if not self.enabled:
            return True
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        if host not in self._cache:
            robots_url = f"{parsed.scheme}://{host}/robots.txt"
            allow_all = False
            parser = None
            try:
                response = self.client.get(robots_url, allow_404=True)
            except ScrapeError:
                allow_all = True  # nicht erreichbar -> wie RFC: erlauben
            else:
                if response.status_code in (404, 410):
                    allow_all = True
                elif response.status_code in (401, 403):
                    allow_all = False
                else:
                    try:
                        body = response.content.decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(body.splitlines())
                response.close()
            self._cache[host] = (allow_all, parser)

        allow_all, parser = self._cache[host]
        if allow_all:
            return True
        if parser is None:
            return False
        if parser.can_fetch(self.user_agent, url):
            return True
        if host not in self._warned:
            self._warned.add(host)
            print(f"  Hinweis: robots.txt von {host} verbietet {url} - wird uebersprungen.")
        return False


# ── Quellen ────────────────────────────────────────────────────────────────

@dataclass
class ItemRef:
    source_id: str
    url: str
    kind: str = "item"
    title: str = ""
    description: str = ""
    prompt: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    license: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceResult:
    record: dict[str, Any]
    files: list[dict[str, Any]] = field(default_factory=list)


class BaseSource:
    name = "base"
    label = "Basis"
    homepage = ""
    license_note = ""

    def discover(self, client: HttpClient, args: argparse.Namespace) -> list[ItemRef]:
        raise NotImplementedError

    def fetch(self, client: HttpClient, ref: ItemRef, args: argparse.Namespace) -> SourceResult:
        raise NotImplementedError

    def download(
        self,
        client: HttpClient,
        ref: ItemRef,
        result: SourceResult,
        out_dir: Path,
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    # Gemeinsame Basis fuer Metadaten
    def base_record(self, ref: ItemRef) -> dict[str, Any]:
        return {
            "source": self.name,
            "source_id": ref.source_id,
            "url": ref.url,
            "title": ref.title,
            "description": ref.description,
            "prompt": ref.prompt or ref.description or ref.title,
            "prompt_fallback": not bool(ref.description),
            "author": ref.author,
            "tags": list(ref.tags),
            "license": ref.license or self.license_note,
            "files": [],
            "errors": [],
            "terminal": False,
            "ok": False,
            "scraped_at": now_iso(),
        }


# ── Abfielder ──────────────────────────────────────────────────────────────

def clean_description(text: str) -> str:
    """Filtert Credits-/Lizenz-/Werbezeilen, die als Prompt unbrauchbar sind."""
    text = WHITESPACE.sub(" ", str(text or "")).strip()
    lowered = text.lower()
    if len(text) < 8:
        return ""
    noise = (
        "credits",
        "all rights reserved",
        "free download on abfielder",
        "not an official minecraft product",
        "copyright",
        "©",
    )
    if any(marker in lowered for marker in noise):
        return ""
    return text


class AbfielderSource(BaseSource):
    name = "abfielder"
    label = "Abfielder"
    homepage = "https://abfielder.com"
    license_note = "Creator-Angaben in structures_meta.jsonl; keine einheitliche Lizenz"

    PRODUCT_RE = re.compile(r"ProductDetails\.php\?id=(\d+)")
    PREMIUM_RE = re.compile(r"redirectPremium\.php\?id=(\d+)")
    DOWNLOAD_RE = re.compile(r'href="([^"]*ProductDownloadThankYou\.php\?[^"]*)"')
    TOKEN_RE = re.compile(r'href="([^"]*serveProductDownload\.php\?[^"]*)"')

    def discover(self, client: HttpClient, args: argparse.Namespace) -> list[ItemRef]:
        refs: list[ItemRef] = []
        seen: set[str] = set()

        if args.abfielder_sitemap:
            try:
                xml = client.get_text(f"{self.homepage}/sitemap.xml")
            except ScrapeError as exc:
                print(f"  Abfielder-Sitemap nicht lesbar ({exc}) - nutze Browse-Seiten.")
            else:
                for url in LOC_PATTERN.findall(xml):
                    if "/schematicdetail/" not in url or url in seen:
                        continue
                    seen.add(url)
                    refs.append(ItemRef(source_id=f"url:{url}", url=url, kind="detail"))

        start = max(1, args.abfielder_start_page)
        page = start
        max_page = start + args.abfielder_pages - 1 if args.abfielder_pages > 0 else 10**9
        empty_pages = 0
        while page <= max_page:
            url = (
                f"{self.homepage}/Products/BrowseProducts.php"
                f"?page={page}&sort=trending&game=minecraft"
            )
            try:
                page_html = client.get_text(url)
            except ScrapeError as exc:
                print(f"  Abfielder-Seite {page} nicht lesbar: {exc}")
                break
            ids = list(dict.fromkeys(self.PRODUCT_RE.findall(page_html)))
            pages = [int(value) for value in re.findall(r"[?&]page=(\d+)", page_html)]
            last_page = max(pages) if pages else page
            new_ids = [pid for pid in ids if pid not in seen]
            for pid in new_ids:
                seen.add(pid)
                refs.append(
                    ItemRef(
                        source_id=pid,
                        url=f"{self.homepage}/Products/ProductDetails.php?id={pid}",
                    )
                )
            if not new_ids:
                empty_pages += 1
                if empty_pages >= 2:
                    break
            else:
                empty_pages = 0
            if page >= last_page:
                break
            page += 1
        print(f"  Abfielder: {len(refs)} freie Produktseiten gefunden")
        return refs

    def parse_product(self, page_html: str) -> dict[str, Any]:
        """Zerlegt eine Abfielder-Produktseite in Metadaten + Download-Links."""
        title_match = re.search(r"<h1[^>]*>(.*?)</h1>", page_html, re.S | re.I)
        title = html_to_text(title_match.group(1)) if title_match else ""
        og_desc = re.search(
            r'<meta[^>]+(?:name|property)="(?:og:)?description"[^>]+content="([^"]*)"',
            page_html,
            re.I,
        )
        og_text = html_to_text(og_desc.group(1)) if og_desc else ""
        description = og_text
        desc_match = re.search(
            r'<div class="product-description">(.*?)</div>', page_html, re.S | re.I
        )
        if desc_match:
            description = html_to_text(desc_match.group(1)) or description

        author = ""
        match = re.search(r"created by\s+([^<.]+?)(?:\.|,|\s*\|)", og_text, re.I)
        if match:
            author = match.group(1).strip()
        if not author:
            match = re.search(r"\bby\s+([^<|]{1,60}?)\s*\|\s*abfielder", page_html, re.I)
            if match:
                author = match.group(1).strip()

        tags = [
            html_to_text(value)
            for value in re.findall(
                r'class="product-tag"[^>]*>(.*?)</a>', page_html, re.S | re.I
            )
        ]
        tags = [tag for tag in dict.fromkeys(tags) if tag]

        links: list[str] = []
        for raw in self.DOWNLOAD_RE.findall(page_html):
            link = html_module.unescape(raw)
            if "convert=" in link:
                continue
            if link not in links:
                links.append(urllib.parse.urljoin(AbfielderSource.homepage, link))

        versions = list(
            dict.fromkeys(
                re.findall(r"Minecraft\s+(1\.\d+(?:\.\d+)?)", page_html, re.I)
            )
        )
        return {
            "title": title,
            "description": description,
            "author": author,
            "tags": tags,
            "download_links": links,
            "mc_versions": versions,
            "premium": not links,
        }

    def fetch(self, client: HttpClient, ref: ItemRef, args: argparse.Namespace) -> SourceResult:
        page_html, final_url = client.get_text_with_url(ref.url)
        parsed = self.parse_product(page_html)
        product_id = ref.source_id
        match = self.PRODUCT_RE.search(final_url)
        if match:
            product_id = match.group(1)

        title = parsed["title"] or f"Abfielder build {product_id}"
        description = clamp_text(clean_description(parsed["description"]), args.max_prompt_chars)
        record = {
            "source": self.name,
            "source_id": product_id,
            "url": final_url,
            "title": title,
            "description": description,
            "prompt": description or title,
            "prompt_fallback": not bool(description),
            "author": parsed["author"],
            "tags": parsed["tags"],
            "mc_versions": parsed["mc_versions"],
            "license": self.license_note,
            "files": [],
            "errors": [],
            "terminal": False,
            "ok": False,
            "scraped_at": now_iso(),
            "_download_links": parsed["download_links"],
        }
        if parsed["premium"] or not parsed["download_links"]:
            record["terminal"] = True
            record["errors"].append("kein freier Download (Premium/ohne Datei)")
            record["ok"] = True
        ref.title = title
        ref.description = description
        ref.prompt = record["prompt"]
        ref.author = parsed["author"]
        ref.tags = parsed["tags"]
        return SourceResult(record)

    def download(
        self,
        client: HttpClient,
        ref: ItemRef,
        result: SourceResult,
        out_dir: Path,
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        record = result.record
        links = record.get("_download_links") or []
        if record.get("terminal") or not links:
            return []

        files: list[dict[str, Any]] = []
        errors: list[str] = list(record.get("errors") or [])
        for index, link in enumerate(links, start=1):
            temp_dir = out_dir / ".tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{record['source_id']}-{index}.download"
            try:
                thank_html = client.get_text(link)
                tokens = self.TOKEN_RE.findall(thank_html)
                if not tokens:
                    raise ScrapeError("Download-Seite ohne Token (Captcha/Login/Premium?)")
                token_url = urllib.parse.urljoin(
                    self.homepage, html_module.unescape(tokens[0])
                )
                download = client.download(
                    token_url,
                    temp_path,
                    max_bytes=int(args.max_file_mb * 1024 * 1024) if args.max_file_mb > 0 else 0,
                    headers={"Referer": link},
                )
                entry = materialize_download(
                    download=download,
                    source=self.name,
                    source_id=record["source_id"],
                    title=record["title"],
                    prompt=record.get("prompt") or record["title"],
                    out_dir=out_dir,
                    args=args,
                    index=index,
                )
                if entry:
                    files.append(entry)
            except (ScrapeError, StructureConversionError, OSError, ValueError) as exc:
                errors.append(f"{link}: {exc}")
            finally:
                temp_path.unlink(missing_ok=True)
                try:
                    temp_path.parent.rmdir()
                except OSError:
                    pass

        record["errors"] = errors
        result.files = files
        return files


# ── Hugging Face (Hack337/Minecraft-Schematics) ───────────────────────────

class HuggingFaceSource(BaseSource):
    name = "huggingface"
    label = "Hugging Face (Hack337/Minecraft-Schematics)"
    homepage = "https://huggingface.co"
    license_note = "MIT (Dataset-Card Hack337/Minecraft-Schematics)"
    DEFAULT_DATASET = "Hack337/Minecraft-Schematics"

    def __init__(self, dataset: str | None = None) -> None:
        self.dataset = (dataset or self.DEFAULT_DATASET).strip().strip("/")

    @property
    def base(self) -> str:
        return f"https://huggingface.co/datasets/{self.dataset}"

    def _tree(self, client: HttpClient) -> list[dict[str, Any]]:
        url = f"https://huggingface.co/api/datasets/{self.dataset}/tree/main?recursive=true"
        entries: list[dict[str, Any]] = []
        while url:
            response = client.get(url)
            try:
                payload = response.json()
            finally:
                response.close()
            if not isinstance(payload, list):
                raise ScrapeError(f"Unerwartete Tree-Antwort fuer {self.dataset}")
            entries.extend(payload)
            link = response.headers.get("Link", "")
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else ""
        return entries

    def discover(self, client: HttpClient, args: argparse.Namespace) -> list[ItemRef]:
        tree = self._tree(client)
        raw_files = [
            entry for entry in tree
            if entry.get("type") == "file"
            and entry.get("path", "").startswith("schematics/")
            and Path(entry["path"]).suffix.lower() in {".schem", ".schematic", ".litematic"}
        ]
        blueprint_files = [
            entry for entry in tree
            if entry.get("type") == "file"
            and entry.get("path", "").startswith("blueprints/")
            and entry.get("path", "").endswith(".json")
        ]
        print(
            f"  Hugging Face: {len(raw_files)} Rohdateien, "
            f"{len(blueprint_files)} Blueprints im Datensatz"
        )

        mapping: dict[str, dict[str, Any]] = {}
        try:
            raw_mapping = client.get_json(f"{self.base}/resolve/main/schematics_mapping.json")
            if isinstance(raw_mapping, list):
                for entry in raw_mapping:
                    if isinstance(entry, dict) and entry.get("file"):
                        mapping[urllib.parse.unquote(str(entry["file"]))] = entry
        except ScrapeError:
            pass

        blueprint_meta: dict[str, dict[str, Any]] = {}
        if not args.no_blueprints:
            try:
                text = client.get_text(f"{self.base}/resolve/main/blueprints_mapping.jsonl")
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and item.get("filename"):
                        blueprint_meta[str(item["filename"])] = item
            except ScrapeError:
                pass

        raw_stems = {Path(path).stem for path in mapping}
        refs: list[ItemRef] = []
        if not args.blueprints_only:
            for entry in raw_files:
                path = entry["path"]
                filename = urllib.parse.unquote(Path(path).name)
                meta = mapping.get(filename) or mapping.get(Path(filename).name) or {}
                title = str(meta.get("name") or Path(filename).stem).strip()
                description = str(meta.get("description") or "").strip()
                description = clean_description(description)
                refs.append(
                    ItemRef(
                        source_id=f"raw:{path}",
                        url=f"{self.base}/resolve/main/{path}",
                        kind="raw",
                        title=title,
                        description=description,
                        prompt=description or title,
                        license="MIT (Dataset-Card)",
                        extra={"path": path, "filename": filename},
                    )
                )
                raw_stems.add(Path(filename).stem)

        if not args.no_blueprints:
            for entry in blueprint_files:
                path = entry["path"]
                filename = Path(path).name
                if not args.blueprints_only and Path(filename).stem in raw_stems:
                    continue  # Rohdatei ist vorhanden und wird bereits verarbeitet
                meta = blueprint_meta.get(filename, {})
                prompt = str(meta.get("short_name_en") or "").strip()
                title = prompt or Path(filename).stem.replace("_", " ")
                refs.append(
                    ItemRef(
                        source_id=f"blueprint:{path}",
                        url=f"{self.base}/resolve/main/{path}",
                        kind="blueprint",
                        title=title,
                        description="",
                        prompt=prompt or title,
                        license="MIT (Dataset-Card)",
                        extra={"path": path, "filename": filename},
                    )
                )
        return refs

    def fetch(self, client: HttpClient, ref: ItemRef, args: argparse.Namespace) -> SourceResult:
        record = self.base_record(ref)
        record["license"] = ref.license or self.license_note
        record["blueprint"] = ref.kind == "blueprint"
        record["_hf_path"] = ref.extra.get("path", "")
        return SourceResult(record)

    def download(
        self,
        client: HttpClient,
        ref: ItemRef,
        result: SourceResult,
        out_dir: Path,
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        record = result.record
        temp_dir = out_dir / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{hashlib.sha1(ref.source_id.encode()).hexdigest()[:12]}.download"
        errors: list[str] = []
        files: list[dict[str, Any]] = []
        try:
            if ref.kind == "blueprint":
                payload = client.get_json(ref.url)
                target = out_dir / f"{build_stem(self.name, ref.source_id, record['title'])}.schem"
                stats = blueprint_to_schem(
                    payload, target, name=record["title"], author=record.get("author", "")
                )
                entry = file_entry(
                    target, out_dir, source_format="blueprint", stats=stats, args=args
                )
                if entry:
                    duplicate = False
                    if args.dedupe_by_hash:
                        owner = args._hash_index.get(entry["sha256"])
                        if owner and owner != ref.source_id:
                            target.unlink(missing_ok=True)
                            files.append(
                                {
                                    "status": "duplicate",
                                    "duplicate_of": owner,
                                    "path": entry["path"],
                                    "file": entry["file"],
                                    "bytes": 0,
                                }
                            )
                            duplicate = True
                        else:
                            args._hash_index.setdefault(entry["sha256"], ref.source_id)
                    if not duplicate:
                        sidecar = target.with_suffix(".txt")
                        write_prompt_sidecar(
                            sidecar,
                            sidecar_prompts(
                                record["title"],
                                record.get("prompt") or record["title"],
                                with_name=args.name_prompt,
                            ),
                        )
                        entry["sidecar"] = relative_path(sidecar, out_dir)
                        files.append(entry)
            else:
                download = client.download(
                    ref.url,
                    temp_path,
                    max_bytes=(
                        int(args.max_file_mb * 1024 * 1024)
                        if args.max_file_mb > 0
                        else 0
                    ),
                )
                entry = materialize_download(
                    download=download,
                    source=self.name,
                    source_id=ref.source_id,
                    title=record["title"],
                    prompt=record.get("prompt") or record["title"],
                    out_dir=out_dir,
                    args=args,
                    index=1,
                )
                if entry:
                    files.append(entry)
        except (ScrapeError, StructureConversionError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
        finally:
            temp_path.unlink(missing_ok=True)
            try:
                temp_path.parent.rmdir()
            except OSError:
                pass
        record["errors"] = errors
        result.files = files
        return files


SOURCES: dict[str, type[BaseSource]] = {
    AbfielderSource.name: AbfielderSource,
    HuggingFaceSource.name: HuggingFaceSource,
}


# ── Download -> Zieldatei ──────────────────────────────────────────────────

def file_entry(
    target: Path,
    out_dir: Path,
    *,
    source_format: str,
    stats: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Validiert die erzeugte Datei und baut den Metadaten-Eintrag."""
    if not target.exists() or target.stat().st_size == 0:
        raise ScrapeError(f"Zieldatei fehlt: {target.name}")
    raw = target.read_bytes()
    fmt = detect_structure_format(raw)
    if fmt not in ("schem", "schematic"):
        raise ScrapeError(f"{target.name}: kein gueltiges Schematic ({fmt})")
    size = read_structure_size(raw)
    if size is None or any(value <= 0 for value in size):
        raise ScrapeError(f"{target.name}: Schematic ohne gueltige Dimensionen")
    if max(size) > 32767:
        raise ScrapeError(f"{target.name}: Dimension {size} ist zu gross")
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "path": relative_path(target, out_dir),
        "file": target.name,
        "format": "schem" if fmt == "schem" else "schematic",
        "source_format": source_format,
        "size": list(size),
        "bytes": target.stat().st_size,
        "sha256": digest,
        "converted": source_format != "schem",
        "blocks": int(getattr(stats, "blocks", 0) or 0),
        "palette_size": int(getattr(stats, "palette_size", 0) or 0),
    }


def extract_zip_structure(zip_path: Path, temp_dir: Path) -> Path:
    """Erstes .schem/.schematic/.litematic aus einem ZIP holen."""
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).suffix.lower() in {".schem", ".schematic", ".litematic"}
            and not name.endswith("/")
        ]
        if not candidates:
            raise ScrapeError("ZIP ohne .schem/.litematic")
        member = candidates[0]
        target = temp_dir / f"zip-{hashlib.sha1(member.encode()).hexdigest()[:10]}{Path(member).suffix}"
        with archive.open(member) as source, target.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        return target


def materialize_download(
    *,
    download: Download,
    source: str,
    source_id: str,
    title: str,
    prompt: str = "",
    out_dir: Path,
    args: argparse.Namespace,
    index: int,
) -> dict[str, Any] | None:
    """Konvertiert einen Download in eine .schem/.schematic und schreibt das Sidecar."""
    raw = download.path.read_bytes()
    fmt = detect_structure_format(raw)
    stem = build_stem(source, source_id, title)
    if index > 1:
        stem = f"{stem}-{index}"

    source_suffix = Path(download.filename).suffix.lower()
    if fmt == "zip":
        temp_dir = out_dir / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        extracted = extract_zip_structure(download.path, temp_dir)
        try:
            nested = Download(
                extracted,
                extracted.name,
                "application/octet-stream",
                extracted.stat().st_size,
                hashlib.sha256(extracted.read_bytes()).hexdigest(),
            )
            return materialize_download(
                download=nested,
                source=source,
                source_id=source_id,
                title=title,
                prompt=prompt,
                out_dir=out_dir,
                args=args,
                index=index,
            )
        finally:
            extracted.unlink(missing_ok=True)

    if fmt == "litematic":
        target = out_dir / f"{stem}.schem"
        stats = litematic_to_schem(
            download.path, target, max_voxels=args.max_voxels
        )
        if args.keep_litematic:
            keep = out_dir / f"{stem}.litematic"
            if not keep.exists() or args.redownload:
                shutil.copyfile(download.path, keep)
        source_format = "litematic"
    elif fmt in ("schem", "schematic"):
        suffix = ".schem" if fmt == "schem" else ".schematic"
        target = out_dir / f"{stem}{suffix}"
        if target.exists() and not args.redownload:
            pass  # Resume auf Dateiebene
        else:
            target.write_bytes(_normalise_schem_bytes(raw, fmt))
        source_format = fmt
        size = read_structure_size(target.read_bytes())
        if size and args.max_voxels > 0 and size[0] * size[1] * size[2] > args.max_voxels:
            target.unlink(missing_ok=True)
            raise StructureTooLarge(size, args.max_voxels, target.name)
        stats = None
    else:
        raise ScrapeError(
            f"nicht unterstuetztes Format {fmt!r} (Datei {download.filename or source_suffix})"
        )

    entry = file_entry(target, out_dir, source_format=source_format, stats=stats, args=args)
    if entry is None:
        return None

    # Dedupe optional: identische Zieldatei aus einem anderen Build ueberspringen.
    if args.dedupe_by_hash:
        owner = args._hash_index.get(entry["sha256"])
        if owner and owner != source_id:
            target.unlink(missing_ok=True)
            return {
                "status": "duplicate",
                "duplicate_of": owner,
                "path": entry["path"],
                "file": entry["file"],
                "bytes": 0,
            }
        args._hash_index.setdefault(entry["sha256"], source_id)

    sidecar = target.with_suffix(".txt")
    write_prompt_sidecar(
        sidecar,
        sidecar_prompts(title, prompt or title, with_name=args.name_prompt),
    )
    entry["sidecar"] = relative_path(sidecar, out_dir)
    return entry


def _normalise_schem_bytes(raw: bytes, fmt: str) -> bytes:
    """Stellt sicher, dass die Datei gzip-komprimiert ist (Dataset erwartet das)."""
    if raw[:2] == b"\x1f\x8b":
        return raw
    return gzip.compress(raw, compresslevel=6)


# ── Persistenz ─────────────────────────────────────────────────────────────

def load_meta(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / META_FILENAME
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("source_id")
            if key:
                records[str(key)] = record
    return records


def append_meta(out_dir: Path, record: dict[str, Any]) -> None:
    clean = {key: value for key, value in record.items() if not key.startswith("_")}
    with (out_dir / META_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
        handle.flush()


def collect_prompts(records: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    prompts: dict[str, list[str]] = {}
    for record in records:
        title = record.get("title") or ""
        prompt = record.get("prompt") or ""
        blocks = sidecar_prompts(title, prompt, with_name=True)
        if not blocks:
            continue
        for entry in record.get("files") or []:
            if entry.get("status") == "duplicate":
                continue
            name = entry.get("file") or Path(entry.get("path", "")).name
            if name:
                prompts[name] = blocks
    return prompts


def write_prompts_json(out_dir: Path, records: Iterable[dict[str, Any]]) -> int:
    prompts = collect_prompts(records)
    path = out_dir / PROMPTS_FILENAME
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
    return len(prompts)


def record_complete(record: dict[str, Any], out_dir: Path, *, metadata_only: bool) -> bool:
    if record.get("terminal"):
        return True
    if not record.get("ok"):
        return False
    if metadata_only:
        return True
    files = record.get("files") or []
    if not files:
        return bool(record.get("duplicates"))
    for entry in files:
        if entry.get("status") == "duplicate":
            continue
        relative = entry.get("path")
        if not relative:
            return False
        target = out_dir / relative
        if not target.exists() or target.stat().st_size == 0:
            return False
    return True


# ── Ablauf ─────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sammelt Minecraft-Strukturen aus freien Quellen als Trainingsdaten.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="all",
        help=f"Komma-Liste aus: {', '.join(SOURCES)} oder 'all'",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Zielordner (je Quelle ein Unterordner)")
    parser.add_argument("--max-items", type=int, default=0, help="Pro Quelle maximal N neue Eintraege verarbeiten (0 = alle)")
    parser.add_argument("--limit", type=int, default=0, help="Lauf nach N verarbeiteten Eintraegen beenden (0 = kein Limit)")
    parser.add_argument("--delay", type=float, default=1.5, help="Pause zwischen Requests in Sekunden")
    parser.add_argument("--jitter", type=float, default=0.8, help="Zufaelliger Zusatz zur Pause")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP-Timeout in Sekunden")
    parser.add_argument("--retries", type=int, default=4, help="Wiederholungen bei 429/5xx/Cloudflare")
    parser.add_argument("--max-file-mb", type=float, default=250.0, help="Groessere Dateien ueberspringen (0 = kein Limit)")
    parser.add_argument("--max-voxels", type=int, default=20_000_000, help="Strukturen ueber N Voxel ueberspringen (0 = kein Limit)")
    parser.add_argument("--order", choices=("source", "shuffle", "reverse"), default="source", help="Reihenfolge der Eintraege")
    parser.add_argument("--metadata-only", action="store_true", help="Nur Metadaten/Prompts, keine Strukturdateien")
    parser.add_argument("--redownload", action="store_true", help="Vorhandene Dateien erneut laden")
    parser.add_argument("--refresh-meta", action="store_true", help="Metadaten vorhandener Eintraege neu laden")
    parser.add_argument("--dedupe-by-hash", action="store_true", help="Identische Zieldateien ueberspringen")
    parser.add_argument("--keep-litematic", action="store_true", help="Original-.litematic neben der .schem behalten")
    parser.add_argument("--name-prompt", action=argparse.BooleanOptionalAction, default=True, help="Namen als ersten Prompt-Block schreiben")
    parser.add_argument("--max-prompt-chars", type=int, default=0, help="Prompt-Laenge kappen (0 = ungekuerzt)")
    parser.add_argument("--ignore-robots", action="store_true", help="robots.txt ignorieren (nicht empfohlen)")
    parser.add_argument("--dry-run", action="store_true", help="Nur Quellen auflisten, nichts herunterladen")
    parser.add_argument("--list-sources", action="store_true", help="Quellen und Hinweise anzeigen")
    parser.add_argument("--rebuild-prompts", action="store_true", help="prompts.json aus structures_meta.jsonl neu schreiben")
    parser.add_argument("--abfielder-pages", type=int, default=0, help="Nur N Browse-Seiten (0 = alle)")
    parser.add_argument("--abfielder-start-page", type=int, default=1, help="Erste Abfielder-Browse-Seite")
    parser.add_argument("--abfielder-sitemap", action=argparse.BooleanOptionalAction, default=False, help="Zusaetzlich Sitemap-Detailseiten aufnehmen")
    parser.add_argument("--hf-dataset", default=HuggingFaceSource.DEFAULT_DATASET, help="Hugging-Face-Datensatz (owner/name)")
    parser.add_argument("--no-blueprints", action="store_true", help="Nur Rohdateien, keine JSON-Blueprints rekonstruieren")
    parser.add_argument("--blueprints-only", action="store_true", help="Hugging Face: nur JSON-Blueprints rekonstruieren")
    return parser.parse_args(argv)


def selected_sources(args: argparse.Namespace) -> list[BaseSource]:
    if args.source.strip().lower() == "all":
        names = list(SOURCES)
    else:
        names = [name.strip().lower() for name in args.source.split(",") if name.strip()]
    unknown = [name for name in names if name not in SOURCES]
    if unknown:
        raise SystemExit(f"Unbekannte Quelle(n): {', '.join(unknown)}. Erlaubt: {', '.join(SOURCES)}")
    result: list[BaseSource] = []
    for name in names:
        if name == HuggingFaceSource.name:
            result.append(HuggingFaceSource(args.hf_dataset))
        else:
            result.append(SOURCES[name]())
    return result


def list_sources() -> None:
    print("Verfuegbare Quellen:\n")
    print(f"  {'abfielder':12} {AbfielderSource.label}")
    print(f"  {'':12} {AbfielderSource.homepage} - tausende freie Builds (.litematic), kein Login")
    print(f"  {'':12} {AbfielderSource.license_note}\n")
    print(f"  {'huggingface':12} {HuggingFaceSource.label}")
    print(f"  {'':12} https://huggingface.co/datasets/{HuggingFaceSource.DEFAULT_DATASET}")
    print(f"  {'':12} MIT-Datensatz: Rohdateien + JSON-Blueprints\n")
    print("Bewusst nicht eingebaut (Login, Paywall, KI-Crawler-Verbot oder /api-Sperre):")
    print("  planetminecraft.com, minecraft-schematics.com, mineschematic.com,")
    print("  shulkr.com, bloxelizer.com, grabcraft.com")


def run_source(
    source: BaseSource,
    args: argparse.Namespace,
    client: HttpClient,
    robots: RobotsPolicy,
    *,
    remaining: int,
) -> tuple[dict[str, int], int, list[dict[str, Any]]]:
    out_dir = Path(args.out_root).expanduser() / source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    if args.rebuild_prompts:
        records = load_meta(out_dir)
        count = write_prompts_json(out_dir, records.values())
        print(f"{source.label}: prompts.json neu geschrieben ({count} Strukturen)")
        return {"new": 0, "skipped": 0, "failed": 0, "files": 0, "bytes": 0}, 0, list(records.values())

    print(f"\n=== {source.label} ===")
    print(f"Ziel: {out_dir}")
    refs = source.discover(client, args)
    if not refs:
        print("  Keine Eintraege gefunden.")
        return {"new": 0, "skipped": 0, "failed": 0, "files": 0, "bytes": 0}, 0, []

    if args.order == "shuffle":
        random.shuffle(refs)
    elif args.order == "reverse":
        refs.reverse()

    counters = {"new": 0, "skipped": 0, "failed": 0, "files": 0, "bytes": 0}
    if args.dry_run:
        print(f"Dry-Run: {len(refs)} Eintraege gefunden (nichts wird geladen).")
        for ref in refs[:10]:
            print(f"  - {ref.source_id}  {ref.title or ref.url}")
        if len(refs) > 10:
            print(f"  ... und {len(refs) - 10} weitere")
        return counters, 0, []

    records = load_meta(out_dir)
    args._hash_index = {}
    if args.dedupe_by_hash:
        for record in records.values():
            for entry in record.get("files") or []:
                if entry.get("sha256"):
                    args._hash_index[entry["sha256"]] = record.get("source_id")

    counters = {"new": 0, "skipped": 0, "failed": 0, "files": 0, "bytes": 0}
    processed = 0
    total = len(refs)
    print(f"Zu pruefen: {total} Eintraege | bereits gespeichert: {len(records)}")
    print("-" * 72)

    for position, ref in enumerate(refs, start=1):
        previous = None if (args.refresh_meta or args.redownload) else records.get(ref.source_id)
        if previous and record_complete(previous, out_dir, metadata_only=args.metadata_only):
            counters["skipped"] += 1
            continue
        if args.max_items > 0 and processed >= args.max_items:
            print(f"Limit fuer {source.name} erreicht ({args.max_items}).")
            break
        if remaining > 0 and processed >= remaining:
            print(f"Gesamtlimit erreicht ({args.limit}).")
            break
        if not robots.allowed(ref.url):
            counters["skipped"] += 1
            continue

        print(f"[{position}/{total}] {ref.source_id[:48]} :: ", end="", flush=True)
        try:
            result = source.fetch(client, ref, args)
            record = result.record
            key = str(record.get("source_id") or ref.source_id)
            # Sitemap-Detailseiten werden erst beim Redirect auf die Produkt-ID
            # kanonisiert; dort kann ein bereits geladener Build auftauchen.
            if key != ref.source_id and key in records:
                previous = records[key]
                if record_complete(previous, out_dir, metadata_only=args.metadata_only):
                    counters["skipped"] += 1
                    print("schon vorhanden (kanonische ID)")
                    continue
            if not args.metadata_only and not record.get("terminal"):
                source.download(client, ref, result, out_dir, args)
            record["files"] = [
                entry for entry in result.files if entry.get("status") != "duplicate"
            ]
            duplicates = [entry for entry in result.files if entry.get("status") == "duplicate"]
            record["duplicates"] = duplicates
            record["ok"] = not record.get("errors") and (args.metadata_only or bool(result.files))
            if record.get("terminal"):
                record["ok"] = True
        except (ScrapeError, StructureConversionError, OSError, ValueError, json.JSONDecodeError) as exc:
            record = source.base_record(ref)
            record["ok"] = False
            record["errors"] = [str(exc)]
            key = ref.source_id

        records[key] = record
        append_meta(out_dir, record)
        processed += 1

        fresh = [entry for entry in record.get("files") or [] if entry.get("status") != "duplicate"]
        counters["files"] += len(fresh)
        counters["bytes"] += sum(int(entry.get("bytes") or 0) for entry in fresh)
        if record.get("terminal"):
            counters["skipped"] += 1
            print("uebersprungen :: kein freier Download")
        elif record.get("ok"):
            counters["new"] += 1
            size_mb = sum(int(entry.get("bytes") or 0) for entry in fresh) / 1024 / 1024
            converted = sum(1 for entry in fresh if entry.get("converted"))
            print(
                f"ok :: {len(fresh)} Datei(en) {size_mb:.2f} MB"
                + (f", {converted} konvertiert" if converted else "")
            )
        else:
            counters["failed"] += 1
            error = "; ".join(record.get("errors") or [])[:160]
            print(f"FEHLER: {error}")

        if processed % 25 == 0:
            write_prompts_json(out_dir, records.values())

    final_count = write_prompts_json(out_dir, records.values())
    print("-" * 72)
    print(f"  neu/aktualisiert : {counters['new']}")
    print(f"  uebersprungen    : {counters['skipped']}")
    print(f"  Fehler           : {counters['failed']}")
    print(f"  Dateien          : {counters['files']} ({counters['bytes'] / 1024 / 1024:.1f} MB)")
    print(f"  prompts.json     : {final_count} Strukturen")
    return counters, processed, list(records.values())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_sources:
        list_sources()
        return 0

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    sources = selected_sources(args)

    if args.rebuild_prompts:
        for source in sources:
            run_source(source, args, HttpClient(delay=0, jitter=0, timeout=30, retries=0), RobotsPolicy(None, USER_AGENT, False), remaining=0)
        return 0

    print("Minecraft-Struktur-Scraper (mehrere Quellen)")
    print(f"Quellen   : {', '.join(source.name for source in sources)}")
    print(f"Ziel      : {out_root.resolve()}")
    print(f"Rate-Limit: {args.delay}s + {args.jitter}s Jitter | Retries: {args.retries}")
    print(f"Modus     : {'nur Metadaten' if args.metadata_only else 'Metadaten + Strukturen'}")

    client = HttpClient(
        delay=args.delay,
        jitter=args.jitter,
        timeout=args.timeout,
        retries=args.retries,
    )
    robots = RobotsPolicy(client, USER_AGENT, enabled=not args.ignore_robots)
    if args.ignore_robots:
        print("WARNUNG: --ignore-robots aktiv - die Seitenbetreiber werden nicht respektiert.")

    totals = {"new": 0, "skipped": 0, "failed": 0, "files": 0, "bytes": 0}
    processed_total = 0
    interrupted = False
    try:
        for source in sources:
            remaining = max(0, args.limit - processed_total) if args.limit > 0 else 0
            counters, processed, _ = run_source(
                source, args, client, robots, remaining=remaining
            )
            for key in totals:
                totals[key] += counters[key]
            processed_total += processed
            if args.limit > 0 and processed_total >= args.limit:
                print(f"\nGesamtlimit von {args.limit} Eintraegen erreicht.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nAbgebrochen (Strg+C). Resume: denselben Befehl erneut starten.")
    finally:
        client.close()

    print("\n" + "=" * 72)
    print("Gesamtergebnis")
    print(f"  neue/aktualisierte Strukturen : {totals['new']}")
    print(f"  uebersprungen                 : {totals['skipped']}")
    print(f"  Fehler                        : {totals['failed']}")
    print(f"  Dateien                       : {totals['files']} ({totals['bytes'] / 1024 / 1024:.1f} MB)")
    print(f"  HTTP-Requests                 : {client.request_count} (Retries: {client.retry_count})")
    print(f"  Ausgabe                       : {out_root.resolve()}")
    if not args.metadata_only:
        print(
            "\nAls Trainingsdaten nutzen:"
            f'\n  python train.py --data-dir "{out_root.resolve()}" --model-size 16 --epochs 10'
        )
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
