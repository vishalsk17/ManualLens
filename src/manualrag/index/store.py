"""SQLite persistence with product-scoped reads and atomic ingest commits."""
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import json
import sqlite3

from manualrag.models import Chunk, ErrorCode, Figure, Page, Product

DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS products(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, model TEXT NOT NULL,
 pdf_sha256 TEXT NOT NULL, n_pages INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pages(
 product_id TEXT NOT NULL REFERENCES products(id), page_no INTEGER NOT NULL,
 png_path TEXT NOT NULL, thumb_path TEXT NOT NULL, page_card_json TEXT,
 payload TEXT NOT NULL, PRIMARY KEY(product_id,page_no));
CREATE TABLE IF NOT EXISTS chunks(
 chunk_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
 page_no INTEGER NOT NULL, type TEXT NOT NULL, section_path TEXT NOT NULL,
 text TEXT NOT NULL, embed_text TEXT NOT NULL, bbox TEXT, bbox_source TEXT NOT NULL,
 is_warning INTEGER NOT NULL, figure_id TEXT, ordinal INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_product ON chunks(product_id,page_no);
CREATE TABLE IF NOT EXISTS figures(
 figure_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
 page_no INTEGER NOT NULL, bbox TEXT, crop_path TEXT, caption TEXT,
 callouts_json TEXT, description TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS error_codes(
 product_id TEXT NOT NULL REFERENCES products(id), code_norm TEXT NOT NULL, code_raw TEXT NOT NULL,
 meaning TEXT, cause TEXT, remedy TEXT, page_no INTEGER NOT NULL, chunk_id TEXT NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(product_id,code_norm,chunk_id));
CREATE TABLE IF NOT EXISTS conversations(
 id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id), active_context TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns(
 id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id),
 question TEXT NOT NULL, answer_json TEXT NOT NULL, cited_pages TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces(
 id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, payload TEXT NOT NULL,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""

class Store:
    """Open short-lived SQLite connections so Streamlit threads never share a cursor."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(DDL)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Commit on success and roll back on any failure."""
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_bundle(self, product: Product, pages: list[Page], chunks: list[Chunk],
                    figures: list[Figure] | None = None, errors: list[ErrorCode] | None = None) -> None:
        """Replace the same product atomically without duplicating extracted records."""
        with self.connect() as db:
            db.execute("INSERT INTO products VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,n_pages=excluded.n_pages",
                       (product.id, product.name, product.model, product.pdf_sha256, product.n_pages, product.model_dump_json()))
            for table in ("error_codes", "chunks", "figures", "pages"):
                db.execute(f"DELETE FROM {table} WHERE product_id=?", (product.id,))
            db.executemany("INSERT INTO pages VALUES(?,?,?,?,?,?)",
                           [(p.product_id, p.page_no, p.png_path, p.thumb_path, p.page_card_json, p.model_dump_json()) for p in pages])
            db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           [(c.chunk_id,c.product_id,c.page_no,c.type,c.section_path,c.text,c.embed_text,json.dumps(c.bbox),
                             c.bbox_source,int(c.is_warning),c.figure_id,c.ordinal,c.model_dump_json()) for c in chunks])
            # Explicit column count avoids hidden schema drift in records.
            for f in figures or []:
                db.execute("INSERT INTO figures VALUES(?,?,?,?,?,?,?,?,?)",
                           (f.figure_id,f.product_id,f.page_no,json.dumps(f.bbox),f.crop_path,f.caption,
                            json.dumps(f.callouts),f.description,f.model_dump_json()))
            for e in errors or []:
                db.execute("INSERT INTO error_codes VALUES(?,?,?,?,?,?,?,?,?)",
                           (e.product_id,e.code_norm,e.code_raw,e.meaning,e.cause,e.remedy,e.page_no,e.chunk_id,e.model_dump_json()))

    def products(self) -> list[Product]:
        """List all ingested manuals."""
        with self.connect() as db:
            return [Product.model_validate_json(r[0]) for r in db.execute("SELECT payload FROM products ORDER BY name,model")]

    def product(self, product_id: str) -> Product:
        """Return a product or raise a clear lookup error."""
        with self.connect() as db:
            row = db.execute("SELECT payload FROM products WHERE id=?", (product_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown manual")
        return Product.model_validate_json(row[0])

    def chunks(self, product_id: str, types: set[str] | None = None) -> list[Chunk]:
        """Read chunks from exactly one manual."""
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM chunks WHERE product_id=? ORDER BY page_no,ordinal,chunk_id", (product_id,))
            chunks = [Chunk.model_validate_json(r[0]) for r in rows]
        return [c for c in chunks if types is None or c.type in types]

    def pages(self, product_id: str) -> list[Page]:
        """Read pages in physical PDF order."""
        with self.connect() as db:
            return [Page.model_validate_json(r[0]) for r in db.execute("SELECT payload FROM pages WHERE product_id=? ORDER BY page_no", (product_id,))]

    def page(self, product_id: str, page_no: int) -> Page:
        """Read a page only within the selected product."""
        with self.connect() as db:
            row = db.execute("SELECT payload FROM pages WHERE product_id=? AND page_no=?", (product_id,page_no)).fetchone()
        if row is None:
            raise ValueError("Page is not in this manual")
        return Page.model_validate_json(row[0])

    def figures(self, product_id: str, page_no: int | None = None) -> list[Figure]:
        """Read detected figures scoped to the selected manual."""
        with self.connect() as db:
            result = [Figure.model_validate_json(r[0]) for r in db.execute("SELECT payload FROM figures WHERE product_id=?", (product_id,))]
        return [f for f in result if page_no is None or f.page_no == page_no]

    def errors(self, product_id: str, code_norm: str) -> list[ErrorCode]:
        """Look up an exact normalized code in the selected manual."""
        with self.connect() as db:
            return [ErrorCode.model_validate_json(r[0]) for r in db.execute("SELECT payload FROM error_codes WHERE product_id=? AND code_norm=?", (product_id,code_norm))]
